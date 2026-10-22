// Copyright 2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Eq. (5), with the normalized input-latency objective used by the authors'
// released scheduler. Exact search avoids assuming Monge inequalities for noisy fits.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
extern "C" int loongserve_partition(int n, int m, const int64_t* lengths,
    const int64_t* capacities, const double* models, int* splits) {
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
    for (int i=1;i<=n;++i) for(int k=1;k<=m;++k)
      for(int b=1;b<=i;++b) for(int s=k;s>=1;--s) {
        int j=i-b,l=k-s;
        if(tokens[i]-tokens[j]>slots[k]-slots[l]) break;
        if(!std::isfinite(f[ix(j,l)])) continue;
        double latency=models[3*s]+models[3*s+1]*(tokens[i]-tokens[j])
            +models[3*s+2]*(squares[i]-squares[j]);
        if(!std::isfinite(latency)||latency<=0) return -3;
        double cost=f[ix(j,l)]+latency*(inverse[i]-inverse[j]);
        if(cost<f[ix(i,k)]) {f[ix(i,k)]=cost;prev_i[ix(i,k)]=j;prev_k[ix(i,k)]=l;}
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
