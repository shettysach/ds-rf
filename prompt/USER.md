Choose one motion and an image waypoint on the visible floor for the robot's next action.

Your entire response must be one line in this exact shape:
{"motion":"walk","waypoint_2d":[500,700]}

The motion value must be stand or walk. For walk, waypoint_2d must be two
integers in [0,1000], where [0,0] is the top-left image corner and [1000,1000]
is the bottom-right. Select a point on the ground that the robot should walk to.
For stand, waypoint_2d must be null.

The first character of your response must be { and the last character must be }.
Never write triple backticks or the word json. Do not include extra fields, comments,
explanations, or any text outside the object.
