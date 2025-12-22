"""
20. Decision Tree Learning [HARD]

Write a Python function that implements the decision tree learning algorithm for classification.
The function should use recursive binary splitting based on entropy and information gain to build a decision tree.
It should take a list of examples (each example is a dict of attribute-value pairs) and
a list of attribute names as input, and return a nested dictionary representing the decision tree.

Example:
Input:
examples = [
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
                    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
                    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
                ],
                attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
Output:
{
            'Outlook': {
                'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
                'Overcast': 'Yes',
                'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
            }
        }
Reasoning:
Using the given examples, the decision tree algorithm determines that 'Outlook' is the best attribute to split the data initially.
When 'Outlook' is 'Overcast', the outcome is always 'Yes', so it becomes a leaf node.
In cases of 'Sunny' and 'Rain', it further splits based on 'Humidity' and 'Wind', respectively.
The resulting tree structure is able to classify the training examples with the attributes 'Outlook', 'Temperature', 'Humidity', and 'Wind'

Insights and Theory: 

- Buckle Up, Now I'm gonna teach you Decision Trees really well. 

- How about making decisions by throught the correct factors like IF the weather is good, THEN I'll go to play ELSE I sit at home. 
- This is exactly what decision tree does. 
- By dividing the data into smaller, purer groups and then finally getting you the classification decision that weather something is a Yes/No, 0/1 or other ways that you define. 
- But we first need to know hoe mixed the data is, how do we do it ?
- Entropy enters the scene, it measures how mixed the data is . Mathematically it measures the class probability. 
- Formula of Entropy: H = -[i=1 ∑ k​ (p of i)​log(base 2​)(pi​)]. 
- Now we know how mixed the data is, but how do we measure which split removes the highest amount of uncertainity and gets us closer to the result ? 
- Enters Information Gain, with it's mathematical formula, it tells us how much of the uncertainity will a particular split improve. 
- A high information gain is a good question and the inverse explains itself. 
- Formula, IG(A) = H(parent) − (v∈values(A)) ∑ ​(∣Sv​∣)/(|S|) *​(H(Sv​))
- and that's how you'll decide for your first split but what about others ? 

- We again split the data by values, compute the weighted entropy, subtract from parent entropy and pick the attribute with maximum information gain. 

- Yes, this process is naturally recursive. 

-- HOW THIS ALGORITHM FLOWS, END-TO-END: 

  - Start with the full dataset 
  - If pure -> return labels
  - Else: Compute Entropy, Compute information gain for each attribute and pick best best attribute
  - For each value of the attribute, create subset and recursively build trees 
  - Return the constructed tree. 

- Now, build the algorithm yourself or see my solution if you're done

"""

#SOLUTION:

import numpy as np
import math
from collections import Counter 

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    
    def entropy(examples, target_attr): 
        label_count = Counter(example[target_attr] for example in examples)
        total = len(examples)

        entropy = 0.0 
        for count in label_count.values():
            p = count/total 
            entropy -= p*math.log2(p)

        return entropy

    def information_gain(examples, attr, target_attr):
        total_entropy = entropy(examples, target_attr)
        total = len(examples)

        # Partitioning examples
        subset = {}
        for example in examples:
            key = example[attr]
            if key not in subset:
                subset[key] = []
            subset[key].append(example)

        weighted_entropy = 0.0 

        for subset in subset.values():
            weighted_entropy += (len(subset)/total) * entropy(subset, target_attr)

        return total_entropy - weighted_entropy

    def majority_class(examples, target_attr):
        most_comm = Counter([example[target_attr] for example in examples]).most_common(1)[0][0]
        return most_comm

    def decision_tree_learning(examples, attributes, target_attr):
        # Case 1: all examples have the same label → leaf node
        labels = [example[target_attr] for example in examples]
        if len(set(labels)) == 1:
            return labels[0]

        # Case 2: no attributes left → majority vote
        if not attributes:
            return majority_class(examples, target_attr)

        # Choose attribute with highest information gain
        best_attribute = max(attributes, key=lambda attr: information_gain(examples, attr, target_attr))

        tree = {best_attribute: {}}

        # Get unique values of best attribute
        attribute_values = set(example[best_attribute] for example in examples)

        for value in attribute_values:
            subset = [ex for ex in examples if ex[best_attribute] == value]

            if not subset:
                tree[best_attribute][value] = majority_class(examples, target_attr)
            else:
                remaining_attrs = [attr for attr in attributes if attr != best_attribute]
                tree[best_attribute][value] = decision_tree_learning(
                    subset, remaining_attrs, target_attr
                )

        return tree

    return decision_tree_learning(examples, attributes, target_attr)
