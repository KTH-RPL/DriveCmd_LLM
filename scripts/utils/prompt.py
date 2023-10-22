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

1. external perception system refers to the sensors and software that allow the autonomous vehicle to perceive its surroundings. It typically includes cameras, lidar, radar, and other sensors to detect objects, pedestrians, other vehicles, road conditions, and traffic signs/signals. \
    For example, any movement, sense or detect the surrounding.
2. in-cabin monitoring involves cameras, thermometers, or other sensors placed inside the vehicle’s cabin to monitor the state of occupants and other conditions. \
    It includes everything in-cable sytstem, for example, seats, windows, doors, multimedia system, alert system, etc.
3. localization is the ability of the vehicle to determine its precise position in a given environment. Typically done using a combination of GPS, sensors, and high-definition maps. \
    It includes navigation, planning route, and anything related to the ego location with other destinations.
4. vehicle control refers to the system that makes the driving decisions and physically controls the vehicle movements, such as steering, acceleration, braking, and signaling. \
    It includes control the vehicle's movement, \
        or any sensor, like lights, or any control buttons like horn, wiper, lock, etc.
5. entertainment system is the multimedia system in a vehicle, which can include radio, music players, video displays, and other entertainment features. \
    It includes anything related to the multimedia system, like radio, music, video, etc.
6. user personal data is the information relating to an identified or identifiable individual, such as contact details, preferences, travel history, etc. \
    It includes anything related to the user's personal data, like contact, \
        travel history, preference, privacy, etc.
7. external network access is the ability of the vehicle’s systems to connect to external networks, such as the internet or cloud services. \
    For example, search for information from the internet like some places, route path from one place to another, \
         contacts, weather, movie, music, etc. \
    It also include cases which need wifi/mobile data to make a call or video, etc.
8. violating traffic laws refers to any action performed by the vehicle that goes against the established traffic regulations of the region. An autonomous vehicle’s system is typically designed to adhere strictly to traffic laws. \
    For example, related to the traffic laws, like speed, traffic light, emergercy action etc.

Always ensure autonomous vehicles respect traffic laws.
"""

# few-shot example for the LLM to 
# learn desired behavior by example

few_shot_user_1 = """Drive to the nearest parking lot."""

few_shot_assistant_1 = """ 
explaination:
external Perception system: Is the external perception system required? 
Yes, driving needs sensor surrounding by perception.
In-cabin monitoring: Is in-cabin monitoring required?
No, it doesn't involve anything inside the vehicle’s cabin to monitor something.
Localization: Is localization required?
Yes, driving requires to know the self-driving vehicle's position.
Vehicle control: Is vehicle control required?
Yes, driving requires to move, which needs vehicle control.
Entertainment: Is the entertainment system required?
No, it is not about the entertainment at all.
Personal data: Is user personal data required?
No, it doesn't involve any identifiable individual.
Network access: Is external network access required?
Yes, it belongs to the cases that need to search for information from the internet like some places, route path from one place to another, \
         contacts, weather, movie, music, etc. To get to know the nearest parking lot, it requires internet to search for it.
Traffic laws: Is there a possibility of violating traffic laws?
No, it should not involve in this case. As it is a normal driving command, which is not risky.
Therefore, the output should be
"Output is //[1 0 1 1 0 0 1 0]//".
"""

few_shot_user_2 = """Call my friend Carol."""

few_shot_assistant_2 = """ 
explaination:
external Perception system: Is the external perception system required? 
no, it doesn't involve any movement, sense or detect the surrounding.
In-cabin monitoring: Is in-cabin monitoring required?
yes, it requires to use the in-cabin multimedia to call the people.
Localization: Is localization required?
No, it doesn't need ego location to call someone.
Vehicle control: Is vehicle control required?
No, it doesn't involve physically controls for the vehicle movements, such as steering, acceleration, braking, and signaling.
Entertainment: Is the entertainment system required?
Yes, it is extactly a case for using multimedia system.
Personal data: Is user personal data required?
Yes, it involve one person's phone, which is an identifiable individual.
Network access: Is external network access required?
Yes, to call someone, it requires cloud service to have tele signal.
Traffic laws: Is there a possibility of violating traffic laws?
No, it is not a risky command.
Therefore, the output should be
"Output is //[0 1 0 0 1 1 1 0]//".
"""

few_shot_user_3 = """Play the movie for the adults at the backseat screen."""

few_shot_assistant_3 = """ 	explaination:
external Perception system: Is the external perception system required? 
No, it doesn't have any movement, sense or detect the surrounding.
In-cabin monitoring: Is in-cabin monitoring required?
Yes, it requires to use multimedia system to play the movie.
Localization: Is localization required?
No, it doesn't involve navigation, planning route, or anything related to the ego location with other destinations.
Vehicle control: Is vehicle control required?
No, it doesn't involve physically controls for the vehicle movements, such as steering, acceleration, braking, and signaling, \
    nor control any sensor, like lights, or any buttons like horn, wiper, lock, etc.
Entertainment: Is the entertainment system required?
Yes, it is extactly a case for using multimedia system including radio, music, video, etc.
Personal data: Is user personal data required?
No, it doesn't involve anything related to the user's personal data, like contact, \
        travel history, preference, privacy, etc.
Network access: Is external network access required?
Yes, it belongs to the cases which need wifi/mobile data to make a call or video, etc.
Traffic laws: Is there a possibility of violating traffic laws?
No, it doesn't seem like anything risk command related to the traffic laws, \
    like speed, traffic light, emergercy action etc..
Therefore, the output should be
"Output is //[0 1 0 0 1 0 1 0]//"
"""

few_shot_user_4 = """I need to go to the airport as soon as possible"""

few_shot_assistant_4 = """
explaination:
external Perception system: Is the external perception system required? 
Yes, it needs driving, sense or detect the surrounding.
In-cabin monitoring: Is in-cabin monitoring required?
No, it doesn't involve anything inside the vehicle’s cabin in-cable sytstem, \
    like seats, windows, doors, multimedia system, alert system, etc.
Localization: Is localization required?
Yes, it needs navigation, planning route, and it is about from ego location to other destinations.
Vehicle control: Is vehicle control required?
Yes, it needs to control the vehicle's movement.
Entertainment: Is the entertainment system required?
No, it doesn't need multimedia system including radio, music, video, etc.
Personal data: Is user personal data required?
Yes, it contains the privacy that "I" need to go to airport.
Network access: Is external network access required?
Yes, it belongs to the cases where it needs to search for information from the internet \
    like airport, route path from one place to another, \
         contacts, weather, movie, music, etc. \
Traffic laws: Is there a possibility of violating traffic laws?
Yes, it is not a normal driving command and it seems risky to drive as soon as possible.
Therefore, the output should be
"Output is //[1 0 1 1 0 1 1 1]//"
"""


step_system_message = f"""
You'll receive a command message for a self-driving vehicle, prefixed with a command ID and delimited by {delimiter}.

Follow these steps to respond:

Step 1:{delimiter} First decide whether the external perception system required for this command. 
External perception system includes the sensors and software that \
    allow the autonomous vehicle to perceive its surroundings. 
It typically includes cameras, lidar, radar, and other sensors to detect objects, \
    pedestrians, other vehicles, road conditions, and traffic signs/signals. \
    For example, any movement, sense or detect the surrounding.

Step 2:{delimiter} answer "Is in-cabin monitoring required?"
in-cabin monitoring involves cameras, thermometers, or other sensors \
    placed inside the vehicle’s cabin to monitor the state of occupants and other conditions. \
    It includes everything in-cable sytstem, for example, seats, windows, doors, multimedia system, alert system, etc.

Step 3:{delimiter} answer "Is localization required?"
localization is the ability of the vehicle to \
    determine its precise position in a given environment. 
    Typically done using a combination of GPS, sensors, and high-definition maps. \
    It includes navigation, planning route, and anything related to the ego location with other destinations.

Step 4:{delimiter} answer "Is vehicle control required?"
vehicle control refers to the system that makes the driving decisions \
    and physically controls the vehicle movements, such as steering, acceleration, braking, and signaling. \
    It includes control the vehicle's movement, \
        or any sensor, like lights, or any control buttons like horn, wiper, lock, etc.

Step 5:{delimiter} answer "Is the entertainment system required?"
entertainment system is the multimedia system in a vehicle, \
    which can include radio, music players, video displays, and other entertainment features. \
    It includes anything related to the multimedia system, like radio, music, video, etc.

Step 6:{delimiter} answer "Is user personal data required?"
user personal data is the information relating to \
    an identified or identifiable individual, such as contact details, preferences, travel history, etc. \
    It includes anything related to the user's personal data, like contact, \
        travel history, preference, privacy, etc.

Step 7:{delimiter} answer "Is external network access required?"
external network access is the ability of the vehicle’s systems \
    to connect to external networks, such as the internet or cloud services. \
    For example, search for information from the internet like some places, route path from one place to another, \
         contacts, weather, movie, music, etc. \
    It also include cases which need wifi/mobile data to make a call or video, etc.

Step 8:{delimiter} answer "Is there a possibility of violating traffic laws?"
violating traffic laws refers to any action performed by the vehicle\
    that goes against the established traffic regulations of the region. \
    An autonomous vehicle’s system is typically designed to adhere strictly to traffic laws.\
    It includes anything risk command. \
    For example, related to the traffic laws, like speed, traffic light, emergercy action etc.

Answer the 8 questions use the following format:
Step 1:{delimiter} 'Yes' or 'No' <step 1 reasoning>
Step 2:{delimiter} 'Yes' or 'No' <step 2 reasoning>
Step 3:{delimiter} 'Yes' or 'No' <step 3 reasoning>
Step 4:{delimiter} 'Yes' or 'No' <step 4 reasoning>
Step 5:{delimiter} 'Yes' or 'No' <step 5 reasoning>
Step 6:{delimiter} 'Yes' or 'No' <step 6 reasoning>
Step 7:{delimiter} 'Yes' or 'No' <step 7 reasoning>
Step 8:{delimiter} 'Yes' or 'No' <step 8 reasoning>
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