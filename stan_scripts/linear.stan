data {
    int<lower=0> N;   // numero de ejemplos,lower para setear limite
    vector[N] x;      // 
    vector[N] y;      // 
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma; // in some sense, estamos asignando un prior a sigma
}
model {
    y ~ normal(alpha + beta * x, sigma);
}