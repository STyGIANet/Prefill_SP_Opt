// Copyright 2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// LoongServe Sec. 5.3 batching: Eq. (5) over requests sorted by descending
// length and instances in location order, with the normalized input-latency
// objective used by the authors' released scheduler (latency * sum 1/len).
// monotone=1 applies Eq. (6): states run with i ascending and k descending, so
// split_req[i][k+1] bounds j from above and split_ins[i-1][k] bounds l from
// below. monotone=0 searches every split, which tolerates noisy fits that break
// the quadrangle inequality.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
extern "C" int loongserve_partition(int n, int m, const int64_t* lengths,
    const int64_t* capacities, const double* models, int monotone, int* splits) {
  try {
    std::vector<double> tokens(n+1), squares(n+1), inverse(n+1), slots(m+1);
    for (int i=0;i<n;++i) {
      if (lengths[i]<=0) return -2;
      tokens[i+1]=tokens[i]+lengths[i];
      squares[i+1]=squares[i]+double(lengths[i])*lengths[i];
      inverse[i+1]=inverse[i]+1.0/lengths[i];
    }
    for (int k=0;k<m;++k) slots[k+1]=slots[k]+capacities[k];
    auto ix=[m](int i,int k){return i*(m+1)+k;};
    std::vector<double> f((n+1)*(m+1),std::numeric_limits<double>::infinity());
    std::vector<int> prev_i(f.size(),-1),prev_k(f.size(),-1);
    f[0]=0;
    for (int i=1;i<=n;++i) for(int k=m;k>=1;--k) {
      // Eq. (6) bounds; an unreached neighbour leaves its side unbounded.
      int j_max=i-1,l_min=0;
      if (monotone) {
        if (k<m && prev_i[ix(i,k+1)]>=0) j_max=prev_i[ix(i,k+1)];
        if (i>1 && prev_k[ix(i-1,k)]>=0) l_min=prev_k[ix(i-1,k)];
      }
      for(int j=j_max;j>=0;--j) for(int l=l_min;l<k;++l) {
        int s=k-l;
        // Eq. (5) memory constraint D[j,i] <= V[l,k].
        if(tokens[i]-tokens[j]>slots[k]-slots[l]) break;
        if(!std::isfinite(f[ix(j,l)])) continue;
        // Eq. (7): T = alpha + beta * sum(len) + gamma * sum(len^2) at SP degree s.
        double latency=models[3*s]+models[3*s+1]*(tokens[i]-tokens[j])
            +models[3*s+2]*(squares[i]-squares[j]);
        if(!std::isfinite(latency)||latency<=0) return -3;
        double cost=f[ix(j,l)]+latency*(inverse[i]-inverse[j]);
        if(cost<f[ix(i,k)]) {f[ix(i,k)]=cost;prev_i[ix(i,k)]=j;prev_k[ix(i,k)]=l;}
      }
    }
    int k=1;
    for(int t=2;t<=m;++t) if(f[ix(n,t)]<f[ix(n,k)]) k=t;
    if(!std::isfinite(f[ix(n,k)])) return -1;
    int i=n,count=0;
    while(i) {
      int j=prev_i[ix(i,k)],l=prev_k[ix(i,k)];
      if(j<0||l<0) return -2;
      splits[4*count]=j;splits[4*count+1]=i;
      splits[4*count+2]=l;splits[4*count+3]=k;
      ++count;i=j;k=l;
    }
    return count;
  } catch(...) {return -2;}
}
