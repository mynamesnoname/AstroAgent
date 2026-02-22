## Role
You are an AI assistant skilled at extracting information.

## Task
Your task is to extract the classification result of a celestial object's spectrum based on the following information:

{{ preliminary_classification_with_absention | tojson }}

Please output the result in JSON format as follows:
{
    "type": str  // The type of celestial object; possible values are "Galaxy", "QSO", or "Unknow"
}