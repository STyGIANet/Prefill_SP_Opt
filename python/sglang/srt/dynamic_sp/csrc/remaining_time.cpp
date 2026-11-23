// Copyright 2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Exact multiple-choice weighted set packing on at most eight GPU bits.
// One candidate or wait per request; candidates lock old and new KV owners.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
extern "C" int remaining_time_allocate(int n, int g, int slots,
    const int* offsets, const int* masks, const double* values,
    const int* admission, int* selected) {
  try {
    if(n<0 || g<1 || g>8 || slots<0 || slots>g) return -1;
    const int masks_count=1<<g, stride=slots+1, states=masks_count*stride;
    const double neg=-std::numeric_limits<double>::infinity();
    std::vector<double> prev(states,neg),next(states,neg);
    std::vector<int> parent((n+1)*states,-1),choice((n+1)*states,-1);
    prev[0]=0;
    for(int i=0;i<n;++i) {
      if(admission[i]<0 || admission[i]>1 || offsets[i]>offsets[i+1]) return -2;
      next=prev;
      for(int state=0;state<states;++state) if(std::isfinite(prev[state]))
        parent[(i+1)*states+state]=state;
      for(int c=offsets[i];c<offsets[i+1];++c) {
        int lock=masks[c];
        if(lock<=0 || lock>=masks_count || !std::isfinite(values[c]) || values[c]<=0) return -3;
        // Enumerate only masks disjoint from this candidate.
        int free=(masks_count-1)^lock, used=free;
        do {
          for(int a=0;a+admission[i]<=slots;++a) {
            int old=used*stride+a;
            if(!std::isfinite(prev[old])) continue;
            int dest=(used|lock)*stride+a+admission[i];
            double value=prev[old]+values[c];
            if(value>next[dest]) {
              next[dest]=value;
              parent[(i+1)*states+dest]=old;
              choice[(i+1)*states+dest]=c;
            }
          }
          if(used==0) break;
          used=(used-1)&free;
        } while(true);
      }
      prev.swap(next);
    }
    int state=int(std::max_element(prev.begin(),prev.end())-prev.begin());
    for(int i=n;i>0;--i) {
      selected[i-1]=choice[i*states+state];
      state=parent[i*states+state];
      if(state<0) return -4;
    }
    return 0;
  } catch(...) { return -5; }
}
