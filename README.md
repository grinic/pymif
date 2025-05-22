# pymif

This is a collection of useful functions and workflows in Python that can be used to process/analyze data acquired on some microscopes available at the [Mesoscopic Imaging Facility (MIF)](https://www.embl.org/groups/mesoscopic-imaging-facility/) at EMBL Barcelona .

## Usage

### Copy the repository on your machine

Clone or download the repository in a path location of your machine.

1. Clone:
```
cd existing_repo
git remote add origin https://git.embl.de/grp-mif/image-analysis/pymif.git
git branch -M main
git push -uf origin main
```

1. Download: Use the `Code/Download source code` blue button at the top right of the page.

### Setup the script

Once downloaded, modifiy the `sys.path.append` line in the examples in the `workflows` folder to point to the right location of the pymif code in your machine.

<!-- You have 2 options:

1. Install the package

Navigate in the folder of the repository containing the `setup.py` file.

In the terminal, create a new environment and install the package:

```
conda create -n pymif python=3.9
pip install -e .
``` -->

### Every time you want to run a workflow

<!-- Activate the environment and start using the functions:

```
conda activate pymif
``` -->

Move the scripts in the `workflows` folder in your location of choice, change the parameters and run the script.

### Ask for help

If you run into problems, contact `nicola.gritti@embl.es`.

<!-- ## Quick start

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.embl.de/grp-mif/image-analysis/pymif.git
git branch -M main
git push -uf origin main
```

### Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

### Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

*** -->
