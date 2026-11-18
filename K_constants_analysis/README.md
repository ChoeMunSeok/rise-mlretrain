# $K_d, K_p$ Constants Analysis

- RCCDA 논문 Appendix 30페이지 참고

## 목표

- RCCDA의 Gain($G^{hat}$) 수식 계산을 위해 필요한 $K_d, K_p$를 동적으로 설정
- K_d, K_p를 동적으로 설정하기 위한 이론적 근거 마련

## Optimal $K_d$ & $K_p$

- **Optimal G(t) in RCCDA**
    - $G(t) = f(\theta_{t-1}, \xi_{t+1}) - f(\theta_{t-1} - \eta∇f(\theta_{t-1}, \zeta_t), \xi_t)$
        - $f(\theta_{t-1}, \xi_{t+1}) = loss_{keep}$
        - $f(\theta_{t-1} - \eta∇f(\theta_{t-1}, \zeta_t), \xi_t) = loss_{retrain} =$ $t_{-1}$ 시점에 학습한 모델을 $t_0$ 시점의 데이터 드리프트만큼 가중치를 보정한 모델
        - 그러므로 $G(t) = loss_{keep} - loss_{retrain}$
- **Practical $G^{hat}(t)$ in RCCDA**
    - RCCDA 논문 Appendix 30이지 참고페
    - $G^{hat}(t) = K_p(f (θt−1, ξt) − min_{i∈{0,...,t}} f(θi, ξi))+K_d(f (θ_i, ξ_i)−f(θ_{t−1}, ξ_{t−1}))$
    - $G^{hat} = K_p(loss(t_0) - min(loss)) + K_d(loss(t_0) - loss(t_{-1}))$
- **Optimal $K_d$$, K_p$**
    - Optimal한 경우, 미래 재학습 Loss의 이득을 예측하는 $G^{hat}(t)$와 $G(t)$의 값은 동일해야함
    - 따라서 $K_p(loss(t_0) - min(loss)) + K_d(loss(t_0) - loss(t_{-1}))= G^{hat} = G = loss_{keep} - loss_{retrain}$
    - $K_p(loss(t_0) - min(loss))  = loss_{keep}$
    - $K_d(loss(t_0) - loss(t_{-1})) = loss_{retrain}$
- $K_p = \frac{loss_{keep}}{(loss(t_0) - min(loss))}$
- $K_d = \frac{loss_{retrain}}{(loss(t_0) - loss(t_{-1})}$

## 동적 $K_p, K_d$ 추정

- **$K_d$$, K_p$**
    - Optimal $K_d, K_p$를 계산하려면 $loss_{keep, t + 1}$과 $loss_{retrain, t + 1}$을 알고 있어야함
    - 하지만 $t + 1$시점의 Loss는 미래이므로 알 수 없음
    - 따라서 $t$시점까지 계산해온 $G^{hat}$의 평균이 기반으로 $t + 1$ 시점의 $G^{hat}(t + 1)$과 동일하다고 가정
    - 이 경우, $loss_{keep} =$  $(Avg$ $G^{hat}$  $= loss_{keep} - loss_{retrain}) + PAPE(t_0)$
    - 이 경우, $loss_{retrain} =$  $-$$(Avg$ $G^{hat}$  $= loss_{keep} - loss_{retrain}) - loss_{keep}(t_{-1}))$
    - 따라서 $K_p(t) = \frac{(Avg G^{hat} + PAPE(t_0)}{PAPE(t_0) - loss_{keep}(t_{-1})}$
    - 따라서 $K_d(t) = \frac{-(Avg G^{hat} - loss_{keep}(t_{-1})}{PAPE(t_0) - loss(t_{-1})}$