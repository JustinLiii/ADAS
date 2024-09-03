# BACKUP for Zhipu API and Gemini API implementation
# elif type(client) is genai.GenerativeModel:
    #     response = client.generate_content(
    #         str(msg_list),
    #         generation_config=genai.types.GenerationConfig(
    #             max_output_tokens=4096,
    #             temperature=temperature,
    #             response_mime_type="application/json"
    #         ),
    #     )
    #     json_dict = response
    # elif type(client) is ZhipuAI:
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             {"role": "user", "content": msg},
    #         ],
    #         temperature=temperature, max_tokens=1024, stop=None
    #     )
    #     content = response.choices[0].message.content
    #     content = content.lstrip('```json').rstrip('```')
    #     try:
    #         json_dict = json.loads(content)
    #     except json.decoder.JSONDecodeError as e:
    #         # print(e)
    #         # print(content)
    #         return {}