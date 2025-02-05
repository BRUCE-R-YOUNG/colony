from roboflow import Roboflow  # Import the Roboflow client

rf = Roboflow(api_key="a14bfchqXiktu80vG8QM")
workspaces = rf.workspace()
print(workspaces)

rf = Roboflow(api_key="a14bfchqXiktu80vG8QM")
workspace = rf.workspace("iotbio-rksli")
projects = workspace.projects()
print(projects)
