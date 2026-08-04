Choose one motion for the robot's next action. You may use either a ground
waypoint in the image or a coarse direction.

Your entire response must be one line in exactly one of these shapes:
{"motion":"walk","waypoint_2d":[500,700]}
{"motion":"walk","direction":"forward"}

The motion value must be stand or walk. Prefer waypoint_2d for a visible floor
target: it is two integers in [0,1000], with [0,0] at the image top-left. A
direction is forward, backward, left, or right. For stand use exactly:
{"motion":"stand","waypoint_2d":null}

The first character of your response must be { and the last character must be }.
Never write triple backticks or the word json. Do not include extra fields,
comments, explanations, or any text outside the object.
