# Disable all firewalls
function disable_firewalls() {
    sudo iptables -F
    sudo iptables -X
    sudo iptables -t nat -F
    sudo iptables -t nat -X
    sudo iptables -t mangle -F
    sudo iptables -t mangle -X
    sudo iptables -P INPUT ACCEPT
    sudo iptables -P FORWARD ACCEPT
    sudo iptables -P OUTPUT ACCEPT

    sudo systemctl stop firewalld
    sudo systemctl disable firewalld

    echo "Disabled all firewalls"
    echo
}

disable_firewalls