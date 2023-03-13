
from os import mkdir, environ
from os.path import isdir, join
from tqdm import tqdm

import openai

openai.api_key = environ["OPENAI_APIKEY"]

# Function to send a message to the OpenAI chatbot model and return its response

def get_response(user_input, character, accuser, room, is_imposter):
    
    player_char = "an Imposter" if is_imposter else "not an Imposter"
    
    # EXTRA STRINGS: "Utilize the following format for response: [next_player, response_string]. The value of 'next_player' is the player that you are talking to, and 'response_string' is your response."
    
    message_log = [
        {"role": "system", "content": f"You are a twitch streamer playing a casual game of 'Among Us' with your friends. Use Among Us slang very liberally. There are four characters in this game: Blue, Red, Green, Yellow. You are currently the player {character} and are {player_char}. The room that you are currently located in is {room}. {accuser} is one who is prompting you. Respond to the given prompts the way that your player would respond."}
    ]
    
    message_log.append({"role": "user", "content": user_input})

    # Add a message from the chatbot to the conversation history
    message_log.append(
        {"role": "assistant", "content": "You are a helpful assistant."})
    
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        # The conversation history up to this point, as a list of dictionaries
        messages=message_log,
        # The maximum number of tokens (words or subwords) in the generated response
        max_tokens=3800,
        # The stopping sequence for the generated response, if any (not used here)
        stop=None,
        # The "creativity" of the generated response (higher temperature = more creative)
        temperature=0.9,
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    return response.choices[0].message.content

def get_initial_message(character, room, is_imposter):
    player_char = "an Imposter" if is_imposter else "not an Imposter"
     
    message_log = [
        {"role": "system", "content": f"You are a twitch streamer playing a casual game of 'Among Us' with your friends. Use Among Us slang very liberally. There are four characters in this game: Blue, Red, Green, Yellow. You are currently the player {character} and are {player_char}. The room that you are currently located in is {room}. You have just called a meeting. Respond to the given prompts the way that your player would respond."}
    ]
    
    message_log.append({"role": "user", "content": "Explain to the other players why you have called a meeting."})

    # Add a message from the chatbot to the conversation history
    message_log.append(
        {"role": "assistant", "content": "You are a helpful assistant."})
    
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        # The conversation history up to this point, as a list of dictionaries
        messages=message_log,
        # The maximum number of tokens (words or subwords) in the generated response
        max_tokens=3800,
        # The stopping sequence for the generated response, if any (not used here)
        stop=None,
        # The "creativity" of the generated response (higher temperature = more creative)
        temperature=0.9,
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    return response.choices[0].message.content

def gen_convo_datapoints(n, data_dir='data'):
    if not isdir(data_dir):
        mkdir(data_dir)
        
    with open(join(data_dir, 'gen'), 'w') as file:
        for i in tqdm(range(n)):
            pass
        


# Main function that runs the chatbot
def main():
    is_imposter = True
    room_name = "Communications"
    
    # print(get_initial_message('Yellow', room_name, is_imposter))
    gen_convo_datapoints(1000)
    
    # Save response information alongside character type and prompted question:
    
    


# Call the main function if this file is executed directly (not imported as a module)
if __name__ == "__main__":
    main()

