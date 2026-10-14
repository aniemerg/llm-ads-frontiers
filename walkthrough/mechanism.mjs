// An inspector for a frozen recorded tape, never a new model generation.
export function weights(a, b) {
  if (![a, b].every(x => Number.isFinite(x) && x >= 0) || !Number.isFinite(a + b) || a + b <= 0) {
    throw new Error('Both bids cannot be zero. Choose the separate no-ad baseline for that case.');
  }
  return [a / (a + b), b / (a + b)];
}

export function frozenOutcome(event, a, b) {
  const [lambda] = weights(a, b);
  if (event.branch === 'shared') {
    return { token_id: event.token_id, branch: 'shared', payment_a: 0, payment_b: 0 };
  }
  if (!['A', 'B'].includes(event.branch) || !event.tape || event.candidate_a == null || event.candidate_b == null) {
    throw new Error('A frozen auction event with residual candidates is required.');
  }
  const u = event.tape.contest;
  if (u < lambda) {
    return { token_id: event.candidate_a, branch: 'A', payment_a: b * (u / (1 - u)), payment_b: 0 };
  }
  return { token_id: event.candidate_b, branch: 'B', payment_a: 0, payment_b: a * ((1 - u) / u) };
}

export const toy = {
  tokens: ['shuttle', 'weekend', 'courtyard'],
  a: [0.6, 0.3, 0.1], b: [0.1, 0.3, 0.6],
  // The supplied Python reference fixture. Both residual supports are singletons.
  event: { token_id: 0, branch: 'A', candidate_a: 0, candidate_b: 2,
    shared_mass: 0.5, tape: { branch: 0.9, token_a: 0.2, token_b: 0.7, contest: 0.4 } },
};
