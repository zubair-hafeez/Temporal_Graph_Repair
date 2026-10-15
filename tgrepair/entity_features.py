"""Observation-only retrieval and symmetric features for identity restoration."""
from bisect import bisect_left
from collections import Counter, defaultdict
import math

import numpy as np


PAIR_FEATURE_NAMES = [
    'log_min_degree','log_max_degree','degree_ratio','log_min_activities','log_max_activities',
    'activity_jaccard','activity_cosine','separation_proximity','raw_overlap','robust_overlap',
    'centroid_proximity','duration_ratio','shared_event_fraction','log_shared_neighbors',
    'neighbor_jaccard','weighted_neighbor_jaccard','neighbor_containment',
    'attribute_agreement','attribute_mode_agreement','attribute_shared_fields','attribute_coverage',
    'duplicate_start_score','duplicate_end_score','boundary_transition_ratio','attribute_numeric_proximity',
]


class EntityIndex:
    def __init__(self,graph,target_types):
        self.graph=graph
        self.target_types=set(target_types)
        self.event_rows=defaultdict(list)
        self.neighbors=defaultdict(Counter)
        self.neighbor_entities=defaultdict(set)
        self.attrs=defaultdict(lambda:defaultdict(Counter))
        self.activities=defaultdict(Counter)
        self.start_counts=Counter();self.end_counts=Counter();self.kind_counts=Counter();self.type_counts=Counter()
        self.transitions=Counter();self.transition_from=Counter()
        for e in graph.events:
            for o in e.participants:
                if graph.objects[o] not in target_types:continue
                self.event_rows[o].append(e)
                self.activities[o][e.kind]+=1
                self.kind_counts[graph.objects[o],e.kind]+=1
                self.type_counts[graph.objects[o]]+=1
                for key,val in e.attributes.items():self.attrs[o][key][val]+=1
                for p in e.participants:
                    if p!=o:
                        self.neighbors[o][p]+=1
                        self.neighbor_entities[p].add(o)
        self.profile={};durations=[]
        for o,events in self.event_rows.items():
            times=np.asarray([e.time for e in events],dtype=float)
            lo,mid,hi=np.quantile(times,[.1,.5,.9])
            self.profile[o]={'n':len(events),'first':times[0],'last':times[-1],
                             'lo':lo,'mid':mid,'hi':hi,'event_ids':{e.uid for e in events}}
            typ=graph.objects[o]
            first=[e.kind for e in events if e.time==times[0]]
            last=[e.kind for e in events if e.time==times[-1]]
            for kind in first:self.start_counts[typ,kind]+=1/len(first)
            for kind in last:self.end_counts[typ,kind]+=1/len(last)
            groups=[]
            for e in events:
                if groups and groups[-1][0]==e.time:groups[-1][1].append(e.kind)
                else:groups.append((e.time,[e.kind]))
            for (t,a),(u,b) in zip(groups,groups[1:]):
                durations.append(u-t)
                for ka in a:
                    for kb in b:
                        weight=1/(len(a)*len(b))
                        self.transitions[typ,ka,kb]+=weight;self.transition_from[typ,ka]+=weight
        self.scale=max(1.,float(np.median(durations))) if durations else 1.
        self.by_type=defaultdict(list)
        for o in self.profile:self.by_type[graph.objects[o]].append(o)
        self.sorted_views={}
        for typ,ids in self.by_type.items():
            for field in ['first','last','mid']:
                order=sorted(ids,key=lambda o:(self.profile[o][field],o))
                self.sorted_views[typ,field]=(order,[self.profile[o][field] for o in order])
        self.modal=defaultdict(dict);self.attribute_groups=defaultdict(set)
        for o,values in self.attrs.items():
            for key,counts in values.items():
                maximum=max(counts.values())
                modes={v for v,n in counts.items() if n==maximum}
                self.modal[o][key]=modes
                for val in modes:self.attribute_groups[graph.objects[o],key,val].add(o)
        self.attribute_views={}
        for key,ids in self.attribute_groups.items():
            order=sorted(ids,key=lambda o:(self.profile[o]['mid'],o))
            self.attribute_views[key]=(order,[self.profile[o]['mid'] for o in order])

    @staticmethod
    def nearest(view,time,budget):
        ids,times=view
        i=bisect_left(times,time);left,right=i-1,i;result=[]
        while len(result)<budget and (left>=0 or right<len(ids)):
            if right>=len(ids) or (left>=0 and time-times[left]<=times[right]-time):
                result.append(ids[left]);left-=1
            else:result.append(ids[right]);right+=1
        return result

    def candidates(self,o,budget=64):
        if o not in self.profile:return []
        typ=self.graph.objects[o];profile=self.profile[o]
        ids=self.by_type[typ]
        if len(ids)<=budget+1:return sorted(p for p in ids if p!=o)
        overlap=Counter()
        for context in self.neighbors[o]:
            neighbors=self.neighbor_entities[context]
            for p in neighbors:
                if p!=o and self.graph.objects[p]==typ:overlap[p]+=1/max(1,len(neighbors))
        streams=[sorted(overlap,key=lambda p:(-overlap[p],p))]
        for field,pivot in [('first','last'),('last','first'),('mid','mid')]:
            streams.append(self.nearest(self.sorted_views[typ,field],profile[pivot],budget+1))
        for key,modes in self.modal[o].items():
            for val in sorted(modes,key=repr):
                view=self.attribute_views[typ,key,val]
                streams.insert(0,self.nearest(view,profile['mid'],budget+1))
        result=set()
        for rank in range(budget+1):
            for stream in streams:
                if rank<len(stream) and stream[rank]!=o:
                    result.add(stream[rank])
                    if len(result)==budget:return sorted(result)
        return sorted(result)

    def pairs(self,budget=64,anchors=None):
        pairs=set()
        for o in sorted(anchors if anchors is not None else self.profile):
            for p in self.candidates(o,budget):pairs.add(tuple(sorted((o,p))))
        return sorted(pairs)

    def features(self,a,b):
        pa,pb=self.profile[a],self.profile[b]
        na,nb=pa['n'],pb['n'];aa,ab=self.activities[a],self.activities[b]
        kinds=set(aa)|set(ab)
        aj=sum(min(aa[k],ab[k]) for k in kinds)/sum(max(aa[k],ab[k]) for k in kinds)
        cosine=sum(aa[k]*ab[k] for k in kinds)/math.sqrt(sum(v*v for v in aa.values())*sum(v*v for v in ab.values()))
        da,db=max(self.scale,pa['last']-pa['first']),max(self.scale,pb['last']-pb['first'])
        gap=max(0.,max(pa['first'],pb['first'])-min(pa['last'],pb['last']))
        overlap=max(0.,min(pa['last'],pb['last'])-max(pa['first'],pb['first']))/min(da,db)
        robust=max(0.,min(pa['hi'],pb['hi'])-max(pa['lo'],pb['lo']))/max(self.scale,min(pa['hi']-pa['lo'],pb['hi']-pb['lo']))
        ca,cb=self.neighbors[a],self.neighbors[b];keys=set(ca)|set(cb);common=set(ca)&set(cb)
        wmin=sum(min(ca[k],cb[k])/len(self.neighbor_entities[k]) for k in keys)
        wmax=sum(max(ca[k],cb[k])/len(self.neighbor_entities[k]) for k in keys)
        shared_fields=set(self.attrs[a])&set(self.attrs[b]);all_fields=set(self.attrs[a])|set(self.attrs[b])
        agree,mode_agree,numeric,num_count=0.,0.,0.,0
        for field in shared_fields:
            xa,xb=self.attrs[a][field],self.attrs[b][field]
            sa,sb=sum(xa.values()),sum(xb.values())
            agree+=sum((xa[v]/sa)*(xb[v]/sb) for v in set(xa)|set(xb))
            mode_agree+=bool(self.modal[a][field]&self.modal[b][field])
            if all(isinstance(v,(int,float)) for v in set(xa)|set(xb)):
                ma=sum(v*n for v,n in xa.items())/sa;mb=sum(v*n for v,n in xb.items())/sb
                numeric+=1/(1+abs(ma-mb)/max(1.,abs(ma),abs(mb)));num_count+=1
        typ=self.graph.objects[a]
        first_a=[e.kind for e in self.event_rows[a] if e.time==pa['first']]
        first_b=[e.kind for e in self.event_rows[b] if e.time==pb['first']]
        last_a=[e.kind for e in self.event_rows[a] if e.time==pa['last']]
        last_b=[e.kind for e in self.event_rows[b] if e.time==pb['last']]
        start_a=sum(self.start_counts[typ,k]/max(1,self.kind_counts[typ,k]) for k in first_a)/len(first_a)
        start_b=sum(self.start_counts[typ,k]/max(1,self.kind_counts[typ,k]) for k in first_b)/len(first_b)
        end_a=sum(self.end_counts[typ,k]/max(1,self.kind_counts[typ,k]) for k in last_a)/len(last_a)
        end_b=sum(self.end_counts[typ,k]/max(1,self.kind_counts[typ,k]) for k in last_b)/len(last_b)
        transition=[]
        for previous,following in [(last_a,first_b),(last_b,first_a)]:
            for ka in previous:
                for kb in following:
                    prior=(self.kind_counts[typ,kb]+1)/(self.type_counts[typ]+len(kinds))
                    prob=(self.transitions[typ,ka,kb]+prior)/(self.transition_from[typ,ka]+1)
                    transition.append(math.log(max(1e-8,prob/prior)))
        return np.asarray([
            math.log1p(min(na,nb)),math.log1p(max(na,nb)),min(na,nb)/max(na,nb),
            math.log1p(min(len(aa),len(ab))),math.log1p(max(len(aa),len(ab))),aj,cosine,
            1/(1+gap/self.scale),overlap,robust,1/(1+abs(pa['mid']-pb['mid'])/max(da,db)),min(da,db)/max(da,db),
            len(pa['event_ids']&pb['event_ids'])/min(na,nb),math.log1p(len(common)),
            len(common)/max(1,len(keys)),wmin/max(1e-12,wmax),len(common)/max(1,min(len(ca),len(cb))),
            agree/max(1,len(shared_fields)),mode_agree/max(1,len(shared_fields)),len(shared_fields),
            len(shared_fields)/max(1,len(all_fields)),min(start_a,start_b),min(end_a,end_b),
            max(transition),numeric/max(1,num_count)
        ],dtype=np.float32)
