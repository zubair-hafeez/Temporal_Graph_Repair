import gzip
import io
import json
import zipfile

from tgrepair.confirmation_sources import load_bpi2017_confirmation,load_bpi2019_confirmation
from tgrepair.json_stream import iter_top_level_array


def test_streaming_parser_uses_root_array_and_handles_chunk_boundaries():
    data={'metadata':{'events':[{'wrong':1}]},'objects':[{'x':'é'*20}]*3,
          'events':[{'id':i,'value':'some text'} for i in range(21)],'later':True}
    for size in [1,7,31]:
        assert list(iter_top_level_array(io.StringIO(json.dumps(data,ensure_ascii=False)),'events',size))==data['events']


def test_bpi2017_excludes_audited_trace_and_retains_only_explicit_offer_links(tmp_path):
    path=tmp_path/'source.xes.gz'
    text='<log>'
    for i in range(3):
        text+=f'<trace><string key="concept:name" value="app{i}"/><int key="RequestedAmount" value="999"/>'
        for j in range(2):
            text+=f'<event><string key="concept:name" value="Work"/><date key="time:timestamp" value="2020-01-0{i+1}T0{j}:00:00Z"/>'
            text+='<string key="org:resource" value="shared"/><string key="EventID" value="secret"/>'
            if j==0:text+=f'<string key="OfferID" value="offer{i}"/><float key="OfferedAmount" value="100"/>'
            text+='</event>'
        text+='</trace>'
    text+='</log>'
    with gzip.open(path,'wt') as stream:stream.write(text)
    graph,audit=load_bpi2017_confirmation(path,cohort_size=2,audited_traces=1)
    assert len(graph.events)==4 and audit['excluded_offers']==1
    assert sum(graph.objects[o]=='Offer' for e in graph.events for o in e.participants)==2
    assert all(set(e.attributes)<={'OfferedAmount'} for e in graph.events)
    assert all('secret' not in str(e) and 'app' not in e.uid for e in graph.events)
    assert min(e.time for e in graph.events)>1577836800+86400-1


def test_bpi2019_excludes_prefix_objects_and_identifier_attributes(tmp_path):
    objects=[{'id':f'p{i}','type':'PO'} for i in range(3)]+[{'id':f'i{i}','type':'POItem'} for i in range(3)]
    objects += [{'id':'r','type':'Resource'},{'id':'v','type':'Vendor'}]
    events=[]
    for i in [0,1,2,0]:
        events.append({'id':str(len(events)),'type':'Receipt','time':'2020-01-01T00:00:00Z',
                       'relationships':[{'objectId':o} for o in [f'p{i}',f'i{i}','r','v']],
                       'attributes':[{'name':'cPOID','value':f'p{i}'},{'name':'eCumNetWorth','value':100}]})
    path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('log.json',json.dumps({'objects':objects,'events':events}))
    graph,audit=load_bpi2019_confirmation(path,cohort_size=2,audited_events=1)
    assert len(graph.events)==2 and audit['excluded_core_objects']==2
    assert all(e.attributes=={'eCumNetWorth':100} for e in graph.events)
    assert len(graph.objects)==6
