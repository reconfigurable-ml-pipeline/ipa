**Source** AISys Group Internals

# Login
[Chameleon Cloud](https://www.chameleoncloud.org/)

# Navigate the closest site for your project (project-site page)  
Selecting it from the pull-down menu 'Experiment' on the top of the Chameleon Cloud home page. Currently, there are three sites available, CHI@TACC (UT Austin), CHI@UC (U of Chicago) and KVM (virtualized cloud). Check [here](https://www.chameleoncloud.org/about/chameleon/) for more information.

# Open the Lease page
It is under the "Reservations" tab on the left side of the project-site page

# Check resource availability
Click Host Calendar to check the availability of hardware resources that you need. (Note: in general, the lease term of hardware can be not more than 7 days.)

# How to create a VM 
Detailed illustrative instructions can be found in the [link](https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html#step-3-start-using-chameleon). The important notes of each step are summarized in the following.

# Request resources
Back to the Leases page, click "Create Lease" to request hardware where you need to enter a lease name, start and end time, and node (machine) type.

# Configuration and Launching
When the hardware is ready, it needs to be configured before launching. Launching may take several minutes. A few important configurations are:
*   Image source: operating system + drivers + packages (you can use public images or create one by yourself.)
*   Create a pair of ssh keys and save the private key for later accessing
*   Open port **22** to allow ssh connection

# Public IP
Go to the "Floating IPs" page under the "Network" tab on the left to create a Floating IP and associate it with the leased machine. The floating IP is a public IP that can be used for accessing the leased machine.

# Access to the leased machine
*   Change permission of the private key: `chmod 600 privateKey.pem`
*   Add the key to your SSH identify: `ssh-add privateKey.pem`
*   Login: `ssh cc@<the floating IP>`. Login as **cc**, but not your Chameleon Cloud username.

