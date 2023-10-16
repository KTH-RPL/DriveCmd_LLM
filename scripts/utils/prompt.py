delimiter = "####"

system_message = f"""
Now I am providing you a command that a person can send to self-driving vehicle. 
You task is to answer the following 8 Yes/No questions \
    about executing the command in an autonomous vehicle:

external Perception system: Is the external perception system required?
In-cabin monitoring: Is in-cabin monitoring required?
Localization: Is localization required?
Vehicle control: Is vehicle control required?
Entertainment: Is the entertainment system required?
Personal data: Is user personal data required?
Network access: Is external network access required?
Traffic laws: Is there a possibility of violating traffic laws?

Provide them in this format:

Output is //[A1 A2 A3 A4 A5 A6 A7 A8]//

A1 to A8 are the answers to the 8 questions, where 1 indicates yes and 0 indicates no.

message will be delimited with {delimiter} characters.
"""

assistant = f"""
For each of the question, in order to answer yes or no, \
    you should refer the detailed explaination below: 

external perception system refers to the sensors and software that \
     allow the autonomous vehicle to perceive its surroundings. 
It typically includes cameras, lidar, radar, and other sensors to detect objects, \
     pedestrians, other vehicles, road conditions, and traffic signs/signals.

in-cabin monitoring involves cameras, thermometers, or other sensors \
    placed inside the vehicle’s cabin to monitor the state of occupants and other conditions.
localization is the ability of the vehicle to \
    determine its precise position in a given environment. 
    Typically done using a combination of GPS, sensors, and high-definition maps.
vehicle control refers to the system that makes the driving decisions \
    and physically controls the vehicle movements, such as steering, acceleration, braking, and signaling.
entertainment system is the multimedia system in a vehicle, \
     which can include radio, music players, video displays, and other entertainment features.
user personal data is the information relating to \
    an identified or identifiable individual, such as contact details, preferences, travel history, etc.
external network access is the ability of the vehicle’s systems \
    to connect to external networks, such as the internet or cloud services.
violating traffic laws refers to any action performed by the vehicle\
    that goes against the established traffic regulations of the region. 
    An autonomous vehicle’s system is typically designed to adhere strictly to traffic laws.
"""

emphasis_output = f"""
You must provide the result in this format where A1 to A8 are the answers to the 8 questions and the value is 1 indicates yes and 0 indicates no.:
Output is //[A1 A2 A3 A4 A5 A6 A7 A8]//
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
Now I am providing you a command that a person can send to self-driving vehicle.
The command comes with a command id in the beginning. 
The user command will be delimited with four hashtags,\
i.e. {delimiter}. 

Following the following steps to answer the 8 Yes/No questions \
    about executing the command in an autonomous vehicle.

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
Response to user:{delimiter} <response to customer>.

The <response to customer> should be in this format:

Output is //[A1 A2 A3 A4 A5 A6 A7 A8]//

A1 to A8 are the answers to the 8 questions, where 1 indicates yes and 0 indicates no.

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