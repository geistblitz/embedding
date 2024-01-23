#include "bert.h"
#include "ggml.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#ifdef WIN32
#include "winsock2.h"
#include "include_win/unistd.h"
typedef int socklen_t;
#define read _read
#define close _close

#define SOCKET_HANDLE SOCKET
#else
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#define SOCKET_HANDLE int
#endif

std::string receive_string(SOCKET_HANDLE socket) {
    static char buffer[1 << 15] = {0};
    ssize_t bytes_received = recv(socket, buffer, sizeof(buffer), 0);
    return std::string(buffer, bytes_received);
}

void send_floats(SOCKET_HANDLE socket, const std::vector<float> floats) {
    send(socket, (const char *)floats.data(), floats.size() * sizeof(float), 0);
}

int main(int argc, char ** argv) {
    char *model = "models/bge-large-en-v1.5/ggml-model-q4_0.bin";

    bert_ctx * bctx;

    if ((bctx = bert_load_from_file(model)) == nullptr) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model);
        return 1;
    }

    char *msg = "hello, world";

    int n_embd = bert_n_embd(bctx);
    std::vector<float> embeddings = std::vector<float>(n_embd);
    bert_encode(bctx, 6, msg, embeddings.data());

    return 0;
}
