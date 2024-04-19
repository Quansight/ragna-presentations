# Replace the $USER occurrences in this file
# with the username that you logged in

local_root = "./.cache/ragna"
authentication = "ragna.deploy.InJupyterHubAuthentication"
document = "ragna.core.LocalDocument"
source_storages = [
    "ragna.source_storages.RagnaDemoSourceStorage",
    "ragna.source_storages.Chroma",
    "ragna.source_storages.LanceDB",
]
assistants = [
    "ragna.assistants.RagnaDemoAssistant",
    "ragna.assistants.Gpt35Turbo16k",
    "ragna.assistants.Gpt4",
    "local_llm.Mistral7BInstruct",
]

[api]
hostname = "127.0.0.1"
port = 31476
root_path = "/user/$USER/proxy/31476"
url = "https://pycon-tutorial.quansight.dev/user/$USER/proxy/31476"
database_url = "sqlite:///./.cache/ragna/ragna.db"
origins = [
    "https://pycon-tutorial.quansight.dev",
]

[ui]
hostname = "127.0.0.1"
port = 31477
origins = [
    "https://pycon-tutorial.quansight.dev",
]
