| Attack Type       | Optimization Method        | $\alpha$ | $\beta$ | $\gamma$ | $\kappa$ | #Epochs | Attack Success Rate | $L_1$  | $L_2$  | $L_\infty$ |
| ----------------- | -------------------------- | -------- | ------- | -------- | -------- | ------- | ------------------- | ------ | ------ | ---------- |
| Untargeted Attack | Softmax Cross Entropy Loss | $0.001$  | $0$     | $0$      | /        | $10000$ | $1.0$               | $9.42$ | $0.22$ | $0.007$    |
| Untargeted Attack | Softmax Cross Entropy Loss | $0.001$  | $0.01$  | $0$      | /        | $10000$ | $0.83$              | $1.54$ | $0.05$ | $0.001$    |
| Untargeted Attack | Softmax Cross Entropy Loss | $0.001$  | $0$     |          | /        | $10000$ | $0.88$              | $1.22$ | $0.05$ | $0.002$    |
| Untargeted Attack | Softmax Cross Entropy Loss | $0.001$  | $0.01$  |          | /        | $10000$ |                     |        |        |            |
| Untargeted Attack | C&W Attack Loss            | $0.001$  | $0$     | $0$      | $100$    | $10000$ |                     |        |        |            |
| Untargeted Attack | C&W Attack Loss            | $0.001$  | $0.01$  | $0$      | $100$    | $10000$ |                     |        |        |            |
| Untargeted Attack | C&W Attack Loss            | $0.001$  | $0$     |          | $100$    | $10000$ |                     |        |        |            |
| Untargeted Attack | C&W Attack Loss            | $0.001$  | $0.01$  |          | $100$    | $10000$ |                     |        |        |            |
| Targeted Attack   | Softmax Cross Entropy Loss | $0.001$  | $0$     | $0$      | /        | $10000$ |                     |        |        |            |
| Targeted Attack   | Softmax Cross Entropy Loss | $0.001$  | $0.01$  | $0$      | /        | $10000$ |                     |        |        |            |
| Targeted Attack   | Softmax Cross Entropy Loss | $0.001$  | $0$     |          | /        | $10000$ |                     |        |        |            |
| Targeted Attack   | Softmax Cross Entropy Loss | $0.001$  | $0.01$  |          | /        | $10000$ |                     |        |        |            |
| Targeted Attack   | C&W Attack Loss            | $0.001$  | $0$     | $0$      | $100$    | $10000$ |                     |        |        |            |
| Targeted Attack   | C&W Attack Loss            | $0.001$  | $0.01$  | $0$      | $100$    | $10000$ |                     |        |        |            |
| Targeted Attack   | C&W Attack Loss            | $0.001$  | $0$     |          | $100$    | $10000$ |                     |        |        |            |
| Targeted Attack   | C&W Attack Loss            | $0.001$  | $0.01$  |          | $100$    | $10000$ |                     |        |        |            |

