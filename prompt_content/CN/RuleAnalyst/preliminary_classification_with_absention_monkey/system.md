## Role
你是一个善于提取信息的 AI 助手。

## Task
你的任务是根据以下信息，提取出对某天体光谱进行分类的结果。

{{ preliminary_classification_with_absention | tojson }}

请输出 json 格式的结果，格式如下：
{
    'type': str,  # 天体类别，可能的取值为 "Galaxy", "QSO", "Unknow"
}
"""
