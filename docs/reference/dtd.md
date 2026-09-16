# dtd

**An important note on DTD definitions (skip to the bottom for the API)**: strictly speaking, a DTD \(\Psi(t)\) can be defined relative to an event rate \(R(t)\) and a star formation rate \(S(t)\). The definition for the DTD is such that its convolution with the SAD gives the event rate:

\[
R(t)=(M * \Psi)(t) = \int_{0}^{t}{S(\tau) \Psi(t - \tau)d \tau}
\]

The trick observers employ to recovering the DTD is discretizing this integral in time, and evaluating it with the present day rate. When they express this as a sum, they can statistically infer the best value for the DTD in the age bins of their measured star formation rate. This introduces the stellar age distribution \(M(t)\), the integral of \(S(t)\), into the formulation. \(M(t)\) is the total stellar mass formed in the time bin. The discretized equation becomes:

\[
R(t_0) = \sum_{j=0}^{t_0}{M_j \Psi_j}
\]

This means that \(\Psi(t)\) has units of events \({\rm yr}^{-1} M_\odot^{-1}\). A DTD can therefore be understood very simply in the context of our BPS models. Call the total mass sampled to create our population of binaries \(M_{\rm sample}\). The time after the burst in which \(M_{\rm sample}\) was formed is just each binary's `tphys`, we will call this \(t_{\textrm{delay}}\). In order to get a DTD, we discretize \(t_{\rm delay}\) into bins. In the jth time bin, we count the number of events which occur and call this \(N_{\rm events,j}\). We say that this jth time has a duration \(\Delta t_{\rm j}\). We find then that the height of the bin \(\Psi_j\) is:

\[
\Psi_j = \frac{N_{\rm events,j}}{\Delta t_{\rm j}  M_{\text{sample}}}
\]

A complication arises immediately when we consider how a DTD could be reconstructed for observations. If an observer wants to understand the timescales over which Wolf-Rayet (WR) stars form, they wont just count WR formations. This is a very difficult task and they are rare, so this would take an implausibly long time for a real result. Rather, they will apply the DTD recovery technique in a slightly different way. If observers want to recover a DTD of WR stars, they count WR stars. When the observer does this counting, they are fundamentally also measuring something which is a function of WR lifetimes, they will recover \(\Psi_{\rm WR}T_{\rm WR}\). One issue, however, is that not every WR will live for the same duration. Trying to work backwards from this direction is a very difficult task, and when it comes time to compare to theory, the DTD we defined above is not the most helpful thing. We want our theoretical "DTD" therefore to be weighted by the length of each event (\(T_{\rm dur}\)). The process on our end is largely the same as before. The largest caveat is that when an event spans multiple bins, it contributes a proportional amount of power to each according to how long it lasted in that bin. Instead of counting the number of discrete events in a bin, we now sum durations:

\[
\Psi_j T_{\rm dur} = \sum_{\text{events} \in \text{bin } j} \frac{T_{\rm dur,event}}{T_{{\rm bin},j} \times M_{\text{sample}}}
\]

And this is the thing we care about and would want to plot to compare to observations. This package allows you to do "both kinds" of DTDs.


::: eventlab.dtd