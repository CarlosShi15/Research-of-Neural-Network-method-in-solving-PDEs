Example K=2 subdomains   1 dimension

$u^{NN}=u^{NN}_1\chi_{\Omega_1}+u^{NN}_2\chi_{\Omega_2}$

$u^{NN}_k=\alpha^T_k\phi_k$       where    $\alpha_k=(\alpha_0^{(k)},\alpha_1^{(k)},\cdots,\alpha_{M_k}^{(k)})^T$   $\phi_k=(\phi_0^{(k)},\phi_1^{(k)},\cdots,\phi_{M_k}^{(k)})^T$     $k=1,2$

$\phi_0^{(k)}=1$    $\phi_m^{(k)}=\sigma(\gamma_k((x-x_c^{(k)})^Ta_m^{(k)}+r_m^{(k)}))$   where $m=1,2,\cdots,M_k$  , $\sigma(x)=tanh(x)$ 

$x_{c_1}, R_1 = 0.125, 0.15$       $x_{c_2}, R_2 = 0.625, 0.40$    $M1=M2=5$   for $\Omega_1=(0,1/4)$  $\Omega_2=(1/4,1)$        (specially Set)

$a_m^{(k)},r_m^{(k)}$ are uniformly distributed in the sphere and $[0,R^{(k)}]$ respectively 

Here $\gamma_1,\gamma_2$  is get by **golden-section search** since we have $\gamma_2=\gamma_1\dfrac{R_1}{R_2}\dfrac{M_2}{M_1}$  ，we randomly choose several points in each required domain in expression of $Loss_{MT}(\alpha)$

so we get $\gamma_1$ by $min_{\gamma_1} \ \eta(\gamma_1,\gamma_2=\gamma_1\dfrac{R_1}{R_2}\dfrac{M_2}{M_1}) $     then we can get $\gamma_2$

Here $\eta(\gamma_1,\gamma_2)=min_{\alpha_1,\alpha_2} Loss_{MT}(\alpha)$

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250906155415613.png" alt="image-20250906155415613" style="zoom:40%;" />

the expression is corresponding to the problem 

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250812160528097.png" alt="image-20250812160528097" style="zoom:40%;" />

Finally get alpha by <img src="/Users/syh/Library/Application Support/typora-user-images/image-20250906155909739.png" alt="image-20250906155909739" style="zoom:40%;" />

$i.e.$ For the problem given is   (include：two interior、boundary conditions、 $[u]=h_1$ and $[\beta u'] = h_2$） 

Then we get all the parameters for $u_{NN}$