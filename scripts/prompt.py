# TODO: Remember to change it later, all model should be the same
DEFAULT_SYSTEM_PROMPT = """\
I'll give you a command for a self-driving vehicle. Determine which of these sections the command uses:
Perception, In-cabin monitoring, Localization, Vehicle control, Entertainment, Personal data, Network access, Traffic laws.
Example: "Drive to the nearest train station." -> [1 0 1 1 0 0 1 0]. 
Command: """

OUTPUT_PROMPT = """Based on your understanding, provide the output and explain using the list of tasks. List the output first, followed by the reasoning."""

FORMAT_PROMPT = """List the binary output, starting and ending with `//`. Example: //The output is [1 0 1 1 0 0 1 0]//"""