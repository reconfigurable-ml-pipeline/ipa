1. Install [Go](https://go.dev/dl/)
```bash
# Define the Go version you want to install
GO_VERSION="1.20"

# Download the Go binary
wget https://golang.org/dl/go$GO_VERSION.linux-amd64.tar.gz

# Extract the archive
sudo tar -C /usr/local -xzf go$GO_VERSION.linux-amd64.tar.gz

# Set up Go environment variables
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.zshrc
echo 'export GOPATH=$HOME/go' >> ~/.zshrc
echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.zshrc

# Load the environment variables for the current session
source ~/.zshrc

# Verify the installation
go version

```
3. Install [Kubebuilder](https://book-v1.book.kubebuilder.io/)
