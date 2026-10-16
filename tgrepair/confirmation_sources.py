"""Fixed source adapters for reserved real-log cohorts.

Only schema and previously audited prefixes inform these adapters. Cohorts use
identifier hashes, never repair performance or history completeness. Calling
these functions on reserved records requires the recorded final protocol.
"""
from dataclasses import replace
from datetime import datetime
import gzip
import hashlib
import io
import math
import xml.etree.ElementTree as ET
import zipfile

from .data import Event,Graph,anonymize
from .json_stream import iter_top_level_array

BPI2017_FIELDS={
    'Action','EventOrigin','FirstWithdrawalAmount','NumberOfTerms','Accepted',
    'MonthlyCost','Selected','CreditScore','OfferedAmount',
}
BPI2019_FIELDS={'eCumNetWorth'}
UNKNOWN={'','UNKNOWN','NONE','NULL','NAN'}


def _rank(value,namespace):
    return hashlib.sha256((namespace+':'+value).encode()).hexdigest()


def _list_hash(values):
    return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()


def _traces(path):
    with gzip.open(path,'rb') as stream:
        parser=ET.iterparse(stream,events=('start','end'));_,root=next(parser)
        for event,node in parser:
            if event=='end' and node.tag.rsplit('}',1)[-1]=='trace':
                yield node
                root.clear()


def _trace_id(trace):
    return next(node.attrib['value'] for node in trace
                if node.tag.rsplit('}',1)[-1]!='event' and node.get('key')=='concept:name')


def _xes_value(node):
    kind=node.tag.rsplit('}',1)[-1];value=node.get('value')
    if kind=='boolean':return value.lower()=='true'
    if kind in {'int','float'}:
        number=float(value)
        return number if math.isfinite(number) else None
    return value


def load_bpi2017_confirmation(path,cohort_size=1024,audited_traces=500,seed=91000):
    names=[];excluded=set();excluded_offers=set()
    for i,trace in enumerate(_traces(path)):
        name=_trace_id(trace);names.append(name)
        if i<audited_traces:
            excluded.add(name)
            for event in trace:
                if event.tag.rsplit('}',1)[-1]=='event':
                    excluded_offers.update(a.get('value') for a in event if a.get('key')=='OfferID')
    if len(set(names))!=len(names):raise ValueError('Repeated application trace identifier')
    selected=set(sorted(set(names)-excluded,key=lambda n:_rank(n,'tgrepair_bpi2017_confirmation_v1'))[:cohort_size])
    if len(selected)!=cohort_size:raise ValueError('Insufficient eligible BPI2017 applications')
    graph=Graph('bpi2017_confirmation_native',{},[]);rejected=0
    for trace in _traces(path):
        name=_trace_id(trace)
        if name not in selected:continue
        application='Application:'+name;graph.objects[application]='Application'
        for event in trace:
            if event.tag.rsplit('}',1)[-1]!='event':continue
            raw={a.get('key'):a.get('value') for a in event}
            if raw.get('OfferID') in excluded_offers:
                raise ValueError('Audited offer reappears outside excluded applications')
            participants=[application]
            offer=raw.get('OfferID')
            if offer:
                obj='Offer:'+offer;graph.objects[obj]='Offer';participants.append(obj)
            resource=raw.get('org:resource','')
            if resource.upper() not in UNKNOWN:
                obj='Resource:'+resource;graph.objects[obj]='Resource';participants.append(obj)
            attributes={}
            for a in event:
                if a.get('key') in BPI2017_FIELDS:
                    value=_xes_value(a)
                    if value is not None:attributes[a.get('key')]=value
                    else:rejected+=1
            timestamp=datetime.fromisoformat(raw['time:timestamp'].replace('Z','+00:00'))
            if timestamp.tzinfo is None:raise ValueError('Timezone-free source timestamp')
            kind=raw['concept:name']+'::'+raw.get('lifecycle:transition','')
            graph.events.append(Event(str(len(graph.events)),timestamp.timestamp(),kind,tuple(participants),attributes))
    graph=anonymize(graph,seed)
    return graph,{'cohort_rule':'lowest SHA256-ranked eligible application IDs, all records of selected applications',
                  'ranking_namespace':'tgrepair_bpi2017_confirmation_v1','cohort_size':cohort_size,
                  'audited_trace_count':audited_traces,'excluded_applications':len(excluded),
                  'excluded_offers':len(excluded_offers),'selected_ids_hash':_list_hash(selected),
                  'excluded_ids_hash':_list_hash(excluded),'nonfinite_attributes_dropped':rejected,
                  'visible_event_fields':sorted(BPI2017_FIELDS),'summary':graph.summary(),
                  'limits':['Application trace keys are reference links only, never event attributes.',
                            'Offer links exist only where OfferID was explicitly recorded.',
                            'Native event fields only; no requested amount copied from application metadata.',
                            'BPI2012 and BPI2017 describe the same institution at different times.']}


def _ocel_array(path,name):
    with zipfile.ZipFile(path) as archive:
        files=[n for n in archive.namelist() if n.endswith('.json')]
        if len(files)!=1:raise ValueError('Expected one JSON source in OCEL archive')
        with archive.open(files[0]) as raw,io.TextIOWrapper(raw,encoding='utf-8') as stream:
            yield from iter_top_level_array(stream,name)


def load_bpi2019_confirmation(path,cohort_size=1024,audited_events=2000,seed=91000):
    objects={o['id']:o['type'] for o in _ocel_array(path,'objects')}
    expected={'PO','POItem','Resource','Vendor'}
    if set(objects.values())!=expected:raise ValueError('Unexpected BPI2019 entity schema')
    prefix=[];excluded=set();selected=None;events=[];scanned=0;excluded_crossings=0
    for i,event in enumerate(_ocel_array(path,'events')):
        scanned+=1
        participants=tuple(sorted({r['objectId'] for r in event['relationships']}))
        if any(o not in objects for o in participants):raise ValueError('Unknown OCEL object')
        if i<audited_events:
            prefix.append(event['id'])
            excluded.update(o for o in participants if objects[o] in {'PO','POItem'})
            continue
        if selected is None:
            selected=set(sorted((o for o,t in objects.items() if t=='PO' and o not in excluded),
                                key=lambda o:_rank(o,'tgrepair_bpi2019_confirmation_v1'))[:cohort_size])
            if len(selected)!=cohort_size:raise ValueError('Insufficient eligible purchase orders')
        if not selected.intersection(participants):continue
        if excluded.intersection(participants):
            excluded_crossings+=1;continue
        attributes={}
        for a in event['attributes']:
            if a['name'] in BPI2019_FIELDS:
                value=a['value']
                if isinstance(value,(str,int,float,bool)) and not (isinstance(value,float) and not math.isfinite(value)):
                    attributes[a['name']]=value
        timestamp=datetime.fromisoformat(event['time'].replace('Z','+00:00'))
        if timestamp.tzinfo is None:raise ValueError('Timezone-free source timestamp')
        events.append(Event(event['id'],timestamp.timestamp(),event['type'],participants,attributes))
    if selected is None:raise ValueError('Source ended within audited prefix')
    present={o for e in events for o in e.participants}
    graph=anonymize(Graph('bpi2019_confirmation_native_ocel',{o:objects[o] for o in sorted(present)},events),seed)
    return graph,{'cohort_rule':'lowest SHA256-ranked eligible purchase-order IDs, all remaining records referencing selected orders',
                  'ranking_namespace':'tgrepair_bpi2019_confirmation_v1','cohort_size':cohort_size,
                  'audited_event_count':audited_events,'excluded_core_objects':len(excluded),
                  'excluded_crossing_events':excluded_crossings,'source_events_scanned':scanned,
                  'selected_ids_hash':_list_hash(selected),'excluded_ids_hash':_list_hash(excluded),
                  'prefix_event_ids_hash':_list_hash(prefix),'visible_event_fields':sorted(BPI2019_FIELDS),
                  'summary':graph.summary(),'limits':[
                      'Format-converted real records; native means the released OCEL representation.',
                      'All c-prefixed trace attributes, ID, idx, resource copies and O2O links excluded.',
                      'Audited purchase-order/item objects are excluded; shared vendors and resources may remain.',
                      'eCumNetWorth is recorded event information, not a unique identifier.']}


def structural_view(graph):
    return Graph(graph.name+'_structural',dict(graph.objects),[replace(e,attributes={}) for e in graph.events])
