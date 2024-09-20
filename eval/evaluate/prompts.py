demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
Model response: {model_output}
Extracted answer: 
""".lstrip()


mathverse_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: 
""".lstrip()


mathverse_key_step_extraction_prompt = """
I will give you a detailed solving procedure or a single answer for a math problem.
If it is a procedure, you need to extract the key solution steps and list them accordingly in markdown syntax.
If it is just a single answer, output the answer directly.
Use XML format to wrap your response. Do not answer any other irrelevant content.

Here are examples:
<model_output>
To begin with, it is important to note that the total measure of the angles surrounding point C must equal 360 degrees. This leads us to the equation: \n\n\\[\n\\angle R C M + \\angle R C L + \\angle M C N + \\angle N C L = 360\n\\]\n\nNext, we can substitute the provided angle expressions into this equation, resulting in:\n\n\\[\n(x - 1) + (3x + 5) + 60 + \\angle N C L = 360\n\\]\n\nUpon simplifying this equation, we combine like terms to obtain:\n\n\\[\n4x + 64 + \\angle N C L = 360\n\\]\n\nFrom here, we can isolate \\(\\angle N C L\\) by rearranging the equation, which gives us:\n\n\\[\n\\angle N C L = 360 - 64 - 4x\n\\]\n\\[\n\\angle N C L = 296 - 4x\n\\]\n\nTo determine the value of \\(x\\), we recognize that the sum of angles \\(\\angle R C M\\) and \\(\\angle R C L\\) must equal the sum of angles \\(\\angle M C N\\) and \\(\\angle N C L\\). This leads us to the equation:\n\n\\[\n(x - 1) + (3x + 5) = 60 + (296 - 4x)\n\\]\n\nSimplifying this equation further, we find:\n\n\\[\n4x + 4 = 356 - 4x\n\\]\n\nBy combining like terms, we arrive at:\n\n\\[\n8x = 352\n\\]\n\nSolving for \\(x\\) yields:\n\n\\[\nx = 44\n\\]\n\nSubstituting this value back into our expression for \\(\\angle N C L\\) allows us to calculate:\n\n\\[\n\\angle N C L = 296 - 4(44)\n\\]\n\\[\n\\angle N C L = 296 - 176\n\\]\n\\[\n\\angle N C L = 120\n\\]\n\nThus, the final answer is: †Answer: B. 120
</model_output>
<extracted>
1. Note that the total measure of the angles surrounding point \(C\) must equal 360 degrees:
   \[
   \\angle R C M + \\angle R C L + \\angle M C N + \\angle N C L = 360
   \]

2. Substitute the provided angle expressions into this equation:
   \[
   (x - 1) + (3x + 5) + 60 + \\angle N C L = 360
   \]

3. Combine like terms:
   \[
   4x + 64 + \\angle N C L = 360
   \]

4. Isolate \(\\angle N C L\) by rearranging the equation:
   \[
   \\angle N C L = 360 - 64 - 4x
   \]
   \[
   \\angle N C L = 296 - 4x
   \]

5. Recognize that the sum of angles \(\\angle R C M\) and \(\\angle R C L\) must equal the sum of angles \(\\angle M C N\) and \(\\angle N C L\):
   \[
   (x - 1) + (3x + 5) = 60 + (296 - 4x)
   \]

6. Simplify this equation:
   \[
   4x + 4 = 356 - 4x
   \]

7. Combine like terms:
   \[
   8x = 352
   \]

8. Solve for \(x\):
   \[
   x = 44
   \]

9. Substitute this value back into the expression for \(\\angle N C L\):
   \[
   \\angle N C L = 296 - 4(44)
   \]
   \[
   \\angle N C L = 296 - 176
   \]
   \[
   \\angle N C L = 120
   \]

### Final Answer
Answer: B. 120
</extracted>
<model_output>
2.2
</model_output>
<extracted>
The single answer is 2.2
</extracted>

Here is what you need to extract:
<model_output>
{model_output}
</model_output>
""".lstrip()


mathverse_multi_step_scoring_prompt = """
I will first give you a visual math problem, including the question, ground-truth answer, and then give you a model output containing multiple key solution steps.
You should give the evaluation first, then the average score and the final answer score.
Please think step by step and output the Average score, along with the Final answer score in the end, as described below:
- Average score: Evaluate, based on the given question, answer, diagram, and diagram annotation, whether each solution step is correct in logical
reasoning, visual perception, and numerical computation, with an incorrect score of 0 and a correct score of 1. Then, calculate the average score of multiple steps.
- Final answer score: Match the model's final answer with the ground truth answer, scoring 1 if it matches and 0 if it doesn't.
- If the model output only includes a single step or answer, the Average score and Final answer score are the same.
- Use XML format to wrap your response. The two scores should be just numbers, which can be decimals, but not fractions or redundancies.

Here is the question:
<question>
{question}
</question>
<ground_truth_answer>
{gt}
</ground_truth_answer>
<model_output>
{extraction}
</model_output>
<evaluation>
...
</evaluation>
<average_score>
...
</average_score>
<final_answer_score>
...
</final_answer_score>
""".lstrip()


mathvista_prompt_score = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
""".lstrip()


geo_qa_extract_prompt = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.
Use XML format to wrap your response. The two scores should be just numbers, which can be decimals, but not fractions or redundancies.

Instances:
<model_response>Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)</model_response>
<extracted_answer>(-2, 1)</extracted_answer>

<model_response>at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.</model_response>
<extracted_answer>D</extracted_answer>

<model_response> at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)</model_response>
<extracted_answer>Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)</extracted_answer>

<model_response>As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.</model_response>
<extracted_answer>null</extracted_answer>

<model_response>Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.</model_response>
<extracted_answer>22.3</extracted_answer>

<model_response> have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)</model_response>
<extracted_answer>f(x) = -x^2 - 2x + 1</extracted_answer>

<model_response>{model_output}</model_response>
""".lstrip()


geo_qa_score_prompt = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.
You should give the evaluation first, then the average score and the final answer score.
Use XML format to wrap your response.

Instances: 
<question>Write the set of numbers represented on the number line in interval notation.</question>
<standard_answer>(-2,1]</standard_answer>
<model_answer>Extracted Answer: \\((-2, 1)\\)</model_answer>
<judgement>0</judgement>

<question>As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}</question>
<standard_answer>C</standard_answer>
<model_answer>B:2\u221a{{3}}</model_answer>
<judgement>0</judgement>

<question>Find the domain and range of the function f using interval notation.</question>
<standard_answer>domain: [-4, 0) and range: (-3, 1]</standard_answer>
<model_answer>Range: \\((-4, 1]\\)</model_answer>
<judgement>0</judgement>

<question>As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}</question>
<standard_answer>C</standard_answer>
<model_answer>null</model_answer>
<judgement>0</judgement>

<question>Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n</question>
<standard_answer>A</standard_answer>
<model_answer>\\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1</model_answer>
<judgement>1</judgement>

<question>{question}</question>
<standard_answer>{gt}</standard_answer>
<model_answer>{extraction}</model_answer>
<explanation>...</explanation>
<judgement>...</judgement>
""".lstrip()
