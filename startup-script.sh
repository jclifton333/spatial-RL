# Install git
sudo apt-get update
sudo apt-get --assume-yes install git

# Clone repo 
git clone https://github.com/jclifton333/spatial-RL


# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user

# Install dependencies
cd spatial-RL
git checkout -b gravity origin/gravity
python3 -m pip install -r requirements.txt --user


