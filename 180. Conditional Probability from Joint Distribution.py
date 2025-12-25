"""
Write a Python function that computes the conditional probability P(A|B),
given a joint probability distribution over events A and B. 
The distribution is provided as a dictionary with keys ('A','B'), ('A','B'), ('A','B'), ('A','B'),
where the backtick ` denotes logical NOT.

Example:
Input:
conditional_probability({('A','B'):0.2, ('A','`B'):0.3, ('`A','B'):0.1, ('`A','`B'):0.4})
Output:
0.6667
Reasoning:
P(B)=0.2+0.1=0.3 and P(A∩B)=0.2, so P(A|B)=0.2/0.3=0.6667.

Insights: 

- None, just a simple math to code case

"""

#SOLUTION: 


def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	# Probability of A and B occurring together
    p_A_and_B = joint_distribution.get(('A', 'B'), 0)

    # Total probability of B
    p_B = (
        joint_distribution.get(('A', 'B'), 0) +
        joint_distribution.get(('`A', 'B'), 0)
    )

    # Handle edge case where P(B) = 0
    if p_B == 0:
        return 0.0

    return round(p_A_and_B / p_B, 4)

    return round(p_A_and_B / p_B, 4)
