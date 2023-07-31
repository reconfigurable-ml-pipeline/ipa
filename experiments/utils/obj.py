import os


# set up object storage
def setup_obj_store():
    os.system("sudo umount -l ~/my_mounting_point")
    os.system("cc-cloudfuse mount ~/my_mounting_point")
