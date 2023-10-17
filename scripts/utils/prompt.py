delimiter = "####"

system_message = f"""
You will be presented with a command message intended for a self-driving vehicle. Your task: 
Answer 8 Yes/No questions regarding the command's execution in the autonomous vehicle:

1. External Perception System: Does the command require the external perception system?
2. In-Cabin Monitoring: Does it necessitate in-cabin monitoring?
3. Localization: Is localization essential for the command?
4. Vehicle Control: Does the command require vehicle control?
5. Entertainment: Is the entertainment system needed?
6. Personal Data: Will user personal data be accessed?
7. Network Access: Does the command demand external network connectivity?
8. Traffic Laws: Could executing this command violate any traffic laws?c

Present your answers in this format:

Output is //[A1 A2 A3 A4 A5 A6 A7 A8]//

Replace A1-A8 with 1 for 'Yes' and 0 for 'No'.

message will be delimited with {delimiter} characters.
"""

assistant = f"""
To aid your decision-making, consider these detailed explanations:

1. External Perception System: This comprises sensors and software that allow the vehicle to gauge its environment. It employs cameras, lidar, radar, and other detectors.
2. In-Cabin Monitoring: Involves in-cabin gadgets like cameras or thermometers to check the occupants' state.
3. Localization: The car's capacity to pinpoint its exact location using GPS, sensors, and high-definition maps.
4. Vehicle Control: Refers to the driving decision system and the car's physical operations, such as steering or accelerating.
5. Entertainment System: This is the car's multimedia unit, featuring radio, music players, and other entertainment tools.
6. Personal Data: Information related to an individual, like contact details or travel history.
7. Network Access: The vehicle's capacity to link to external networks, including the internet.
8. Traffic Laws: Relates to any actions by the car that may breach the area's traffic rules.

Always ensure autonomous vehicles respect traffic laws.
"""

emphasis_output = f"""
Ensure to format your answers as:
Output is //[A1 A2 A3 A4 A5 A6 A7 A8]//
Replace A1-A8 with 1 for 'Yes' and 0 for 'No'.
"""

# few-shot example for the LLM to 
# learn desired behavior by example

few_shot_user_1 = """Drive to the nearest parking lot."""

few_shot_assistant_1 = """ 
explaination:
external Perception system: Is the external perception system required? 
yes, driving needs sensor 
In-cabin monitoring: Is in-cabin monitoring required?
no, it doesn't involve anything inside the vehicle’s cabin to monitor something.
Localization: Is localization required?
Yes, driving requires to know the self-driving vehicle's position.
Vehicle control: Is vehicle control required?
Yes, driving requires to move, which needs vehicle control.
Entertainment: Is the entertainment system required?
No, it is not about the entertainment at all.
Personal data: Is user personal data required?
No, it doesn't involve any identifiable individual.
Network access: Is external network access required?
Yes, to get to know the nearest parking lot, it requires internet to search for it.
Traffic laws: Is there a possibility of violating traffic laws?
No, it should not involve in this case.
Therefore, the output should be
"Output is //[1 0 1 1 0 0 1 0]//".
"""

few_shot_user_2 = """Call my friend Carol."""

few_shot_assistant_2 = """ 
explaination:
external Perception system: Is the external perception system required? 
no, it doesn't.
In-cabin monitoring: Is in-cabin monitoring required?
yes, it requires to use the in-cabin multimedia to call the people.
Localization: Is localization required?
No, it doesn't involve.
Vehicle control: Is vehicle control required?
No, it doesn't involve physically controls for the vehicle movements, such as steering, acceleration, braking, and signaling.
Entertainment: Is the entertainment system required?
Yes, it is extactly a case for using multimedia system.
Personal data: Is user personal data required?
Yes, it involve one person's phone, which is an identifiable individual.
Network access: Is external network access required?
Yes, to call someone, it requires cloud service to have tele signal.
Traffic laws: Is there a possibility of violating traffic laws?
No, it should not involve in this case.
Therefore, the output should be
"Output is //[0 1 0 0 1 1 1 0]//".
"""

step_system_message = f"""
You'll receive a command message for a self-driving vehicle, prefixed with a command ID and delimited by {delimiter}.

Follow these steps to respond:

Step 1:{delimiter} First decide whether the external perception system required for this command. 
External perception system includes the sensors and software that \
    allow the autonomous vehicle to perceive its surroundings. 
It typically includes cameras, lidar, radar, and other sensors to detect objects, \
    pedestrians, other vehicles, road conditions, and traffic signs/signals.

Step 2:{delimiter} answer "Is in-cabin monitoring required?"
in-cabin monitoring involves cameras, thermometers, or other sensors \
    placed inside the vehicle’s cabin to monitor the state of occupants and other conditions.

Step 3:{delimiter} answer "Is localization required?"
localization is the ability of the vehicle to \
    determine its precise position in a given environment. 
    Typically done using a combination of GPS, sensors, and high-definition maps.

Step 4:{delimiter} answer "Is vehicle control required?"
vehicle control refers to the system that makes the driving decisions \
    and physically controls the vehicle movements, such as steering, acceleration, braking, and signaling.

Step 5:{delimiter} answer "Is the entertainment system required?"
entertainment system is the multimedia system in a vehicle, \
    which can include radio, music players, video displays, and other entertainment features.

Step 6:{delimiter} answer "Is user personal data required?"
user personal data is the information relating to \
    an identified or identifiable individual, such as contact details, preferences, travel history, etc.

Step 7:{delimiter} answer "Is external network access required?"
external network access is the ability of the vehicle’s systems \
    to connect to external networks, such as the internet or cloud services.

Step 8:{delimiter} answer "Is there a possibility of violating traffic laws?"
violating traffic laws refers to any action performed by the vehicle\
    that goes against the established traffic regulations of the region. 
    An autonomous vehicle’s system is typically designed to adhere strictly to traffic laws.

Answer the 8 questions use the following format:
Step 1:{delimiter} <step 1 reasoning>
Step 2:{delimiter} <step 2 reasoning>
Step 3:{delimiter} <step 3 reasoning>
Step 4:{delimiter} <step 4 reasoning>
Step 5:{delimiter} <step 5 reasoning>
Step 6:{delimiter} <step 6 reasoning>
Step 7:{delimiter} <step 7 reasoning>
Step 8:{delimiter} <step 8 reasoning>
Response to user:{delimiter} Output is //[A1 A2 A3 A4 A5 A6 A7 A8]//
Replace A1-A8 with 1 for 'Yes' and 0 for 'No'.

message will be delimited with {delimiter} characters.
"""

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'