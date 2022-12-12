from step_6.Environment.Environment import *

class NS_Environment(Environment):

    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon

    def round(self, pulled_arm):
        # 5 phases
        n_phases = len(self.probabilities)
        phase_size = self.horizon/n_phases
        current_phase = int(self.t / phase_size)

        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1
        # Bernoulli distribution over p
        result = np.random.binomial(1,p)
        #print("Probability:", p, " & result:", result)
        return result
