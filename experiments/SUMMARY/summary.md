# Сводка экспериментов

MAPE по 35 прогнозам (5 культур × 2019–2025), правила D8/D10.

```
          model       experiment  mape_tune  mape_eval  eval_ci95_lo  eval_ci95_hi  eval_max_ape  p_vs_baseline_sel  p_vs_anchor_ma3 beats_anchor_on_tune  tune_is_honest
   chronos2_cov     E07-chronos2     22.667     14.326        10.285        17.712        42.373              0.494            0.538                False            True
chronos2_cov_xl     E07-chronos2     22.218     13.186         9.485        16.374        34.079              0.865            0.942                False            True
   chronos2_uni     E07-chronos2     24.054     14.600        12.111        16.947        36.852              0.417            0.234                False            True
      cnn_c_m06  E04-cnn-compact     28.325     14.970        10.154        19.741        59.660              0.459            0.475                False           False
      cnn_c_m07  E04-cnn-compact     26.340     17.039        10.795        24.026        64.262              0.209            0.185                False           False
      cnn_c_m08  E04-cnn-compact     24.452     14.485        10.332        20.013        59.252              0.531            0.572                False           False
      cnn_c_m09  E04-cnn-compact     24.836     13.565        11.061        16.244        57.535              0.791            0.891                False           False
      gru_c_m06  E03-gru-compact     28.032     16.362        10.993        23.779        73.350              0.277            0.270                False           False
      gru_c_m07  E03-gru-compact     28.130     16.646        11.089        22.676        62.355              0.252            0.239                False           False
      gru_c_m08  E03-gru-compact     24.422     17.525        12.936        21.651        58.240              0.131            0.138                False           False
      gru_c_m09  E03-gru-compact     24.166     17.461        13.028        21.705        88.460              0.194            0.146                False           False
        gru_m06 E02-gru-inseason     27.928     20.309        12.624        28.934        58.463              0.023            0.011                False           False
        gru_m07 E02-gru-inseason     24.841     17.530        13.018        22.632        37.783              0.044            0.032                False           False
        gru_m08 E02-gru-inseason     22.470     17.464        13.869        22.040        45.469              0.042            0.036                False           False
        gru_m09 E02-gru-inseason     27.174     15.945        11.006        23.394       132.485              0.460            0.464                False           False
            mlp          E05-mlp     23.079     21.595        17.926        24.698        59.786              0.000            0.002                False           False
   baseline_sel    E01-baselines     16.485     12.839         8.301        17.951        63.253                NaN              NaN                 None           False
     anchor_ma3    E01-baselines     21.495     13.293         9.814        16.321        32.677              0.803              NaN                 None            True
          naive    E01-baselines     24.011     16.155        12.970        18.916        43.775              0.240            0.163                 None            True
          trend    E01-baselines     20.723     12.819         9.996        16.223        38.463              0.992            0.738                 None            True
          ridge    E01-baselines     16.523     13.454         8.522        19.293        63.253              0.170            0.933                 None           False
```
