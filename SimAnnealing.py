# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:00:12 2019

@author: gawe
"""
# Answer the question.  Do I want to write my own simulated annealing
# function or use the scipy implementation of basin hopping?
#     I should use scipy, but how do i get around all the crazy differences
#     on machines!    Maybe Novi, Adrian, and myself just need to pick a
#     single python / numpy / scipy version.
#  I would like to use scikit-learn, but it requires numpy and scipy > I have!

class SimAnnealing(Struct):

    def cost_function(x):
        """ Cost of x = f(x)."""
        return f(x)

    def random_start():
        """ Random point in the interval."""
        a, b = interval
        return a + (b - a) * rn.random_sample()

    def random_neighbour(x, fraction=1):
        """Move a little bit x, from the left or the right."""
        amplitude = (max(interval) - min(interval)) * fraction / 10
        delta = (-amplitude/2.) + amplitude * rn.random_sample()
        return clip(x + delta)

    def clip(x):
        """ Force x to be in the interval."""
        a, b = interval
        return max(min(x, b), a)

    def acceptance_probability(cost, new_cost, temperature):
        if new_cost < cost:
            # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            # print("    - Acceptance probabilty = {:.3g}...".format(p))
            return p

    def temperature(fraction):
        """ Example of temperature dicreasing as the process goes on."""
        return max(0.01, min(1, 1 - fraction))

    def annealing(random_start, cost_function, random_neighbour, acceptance,
                  temperature, maxsteps=1000, debug=True):
        """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
        state = random_start()
        cost = cost_function(state)
        states, costs = [state], [cost]
        for step in range(maxsteps):
            fraction = step / float(maxsteps)
            T = temperature(fraction)
            new_state = random_neighbour(state, fraction)
            new_cost = cost_function(new_state)
            if debug: print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {:>4.3g}, cost = {:>4.3g}, new_state = {:>4.3g}, new_cost = {:>4.3g} ...".format(step, maxsteps, T, state, cost, new_state, new_cost))
            if acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
                # print("  ==> Accept it!")
            # else:
            #    print("  ==> Reject it...")
        return state, cost_function(state), states, costs

    def see_annealing(states, costs):
        plt.figure()
        plt.suptitle("Evolution of states and costs of the simulated annealing")
        plt.subplot(121)
        plt.plot(states, 'r')
        plt.title("States")
        plt.subplot(122)
        plt.plot(costs, 'b')
        plt.title("Costs")
        plt.show()

    def visualize_annealing(cost_function):
        state, c, states, costs = annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=1000, debug=False)
        see_annealing(states, costs)
        return state, c
# end def SimAnnealing

# ========================================================================== #
