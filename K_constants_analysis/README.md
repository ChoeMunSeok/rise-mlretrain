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
    - 다만 $K_p(t)$의 경우, 최종적으로 $loss_{keep}$을 계산하기 위해서는 $loss_{retrain}$이 요구됨. 하지만 이는 해당 시점에 알 수 없는 값임.
    - 따라서 $loss_{retrain}$의 값을 가장 최근에 측정(추정)한 loss인 $PAPE(t_0)$로 치환하여 사용
    - 이 경우, $loss_{keep} =$  $(Avg$ $G^{hat}$  $= loss_{keep} - loss_{retrain}) + PAPE(t_0)$
    - 이 경우, $loss_{retrain} =$  $-(Avg$ $G^{hat}$  $= loss_{keep} - loss_{retrain}) - PAPE(t_0)$
    - 따라서 $K_p(t) = \frac{Avg G^{hat} + PAPE(t_0)}{PAPE(t_0) - loss_{min}}$
    - 따라서 $K_d(t) = \frac{PAPE(t_0) - Avg G^{hat}}{PAPE(t_0) + loss(t_{-1})}$

## Gain estimator 코드
```python
def K_p_fomula(avg_gain, cur_loss, min_loss):
    return (avg_gain + cur_loss) / (cur_loss - min_loss)

def K_d_fomula(avg_gain, cur_loss, prev_loss):
    return (cur_loss - avg_gain) / (cur_loss + prev_loss)

def gain_estimate(cur_loss, min_loss, prev_loss, kp, kd):
    est_loss_keep = kp * (cur_loss - min_loss)
    est_loss_retrain = kd * (cur_loss - prev_loss)
    return est_loss_keep, est_loss_retrain, est_loss_keep + est_loss_retrain
```

## Conclusion
- API graph의 2015, 2016 데이터를 기반으로 Gain 추정 수식 검증
- 검증 결과 $loss_{keep}$과 $loss_{retrain}$은 각각 0.31, 0.36의 피어슨 상관계수를 보여 낮은 수준의 선형 관계를 확인
- 다만 $loss_{keep}$과 $loss_{retrain}$의 합인 $G^{hat}$은 $G^{real}$과 0.18의 피어슨 상관 계수를 보여 각각의 항목보다 낮은 추정 성능을 보임
- 그래프 상으로 보았을 때, $K_d$가 $K_p$에 비해 Optimal 값의 경향성 더 잘 추종함에도 오히려 $loss_{retrain}$의 성능이 높음(원인 분석 필요)
- 결과적으로 보았을 때, 본 추정 수식은 약하게 나마 $G^{real}$의 값을 추종하고 있으나, 실제 모델에 적용하기에는 아직 높은 오차를 보임
- 따라서 수식의 개선이 필요함
