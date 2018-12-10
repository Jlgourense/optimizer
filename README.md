#Problem
=======

Solve {#solve .unnumbered}
-----

$$min \quad J \quad  \left\{\begin{array}{c}
         J=\int_{t_{0}}^{t_{1}}g(y_{state},u_{control},t)dt+S(y_{state}(t_{n}) \\
        y'=f(y_{state},u_{control},t) \end{array} \right.$$

Provided {#provided .unnumbered}
--------

$$\arraycolsep=1.4pt\def\arraystretch{2.2} \begin{array}{lll}
\frac{\partial f}{\partial u}& \quad &\frac{\partial g}{\partial u} \\
\frac{\partial f}{\partial y}& \quad &\frac{\partial g}{\partial y} \\
S& \quad &f\\
\end{array}$$
