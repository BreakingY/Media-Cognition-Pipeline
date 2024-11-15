#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>
#include "rtsp_common.h"
extern "C"{
#include "md5.h"
}
int CreateTcpSocket()
{
    int sockfd;
    int on = 1;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        return -1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&on, sizeof(on));
    return sockfd;
}

int CreateUdpSocket()
{
    int sockfd;
    int on = 1;
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        return -1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&on, sizeof(on));
    return sockfd;
}

int ConnectToServer(int sockfd, const char* ip, int port, int timeout)
{
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &server_addr.sin_addr) <= 0) {
        perror("inet_pton");
        return -1;
    }

    // 设置sockfd为非阻塞
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        perror("fcntl(F_GETFL)");
        return -1;
    }

    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl(F_SETFL)");
        return -1;
    }

    // 连接到服务器
    int ret = connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (ret < 0) {
        if (errno != EINPROGRESS) {
            perror("connect");
            return -1;
        }

        // 使用select等待连接完成
        fd_set writefds;
        FD_ZERO(&writefds);
        FD_SET(sockfd, &writefds);
        struct timeval tv;
        if (timeout >= 0) {
            tv.tv_sec = timeout / 1000;
            tv.tv_usec = (timeout % 1000) * 1000;
        }

        ret = select(sockfd + 1, NULL, &writefds, NULL, timeout >= 0 ? &tv : NULL);
        if (ret <= 0) {
            if (ret == 0) {
                fprintf(stderr, "connect timeout\n");
            } else {
                perror("select");
            }
            return -1;
        }
    }

    // 还原sockfd为阻塞
    if (fcntl(sockfd, F_SETFL, flags) < 0) {
        perror("fcntl(F_SETFL)");
        return -1;
    }

    return 0;
}
int BindSocketAddr(int sockfd, const char *ip, int port)
{
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip);
    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(struct sockaddr)) < 0)
        return -1;
    return 0;
}
bool ParseRTSPUrl(const std::string& rtsp_url, RTSPUrlInfo& url_info) {
    // 格式：rtsp://[username:password@]host[:port]
    std::istringstream iss(rtsp_url);
    char delimiter;

    // 检查是否以 "rtsp://" 开头
    if (!(iss >> delimiter) || delimiter != 'r' ||
        !(iss >> delimiter) || delimiter != 't' ||
        !(iss >> delimiter) || delimiter != 's' ||
        !(iss >> delimiter) || delimiter != 'p' ||
        !(iss >> delimiter) || delimiter != ':' ||
        !(iss >> delimiter) || delimiter != '/' ||
        !(iss >> delimiter) || delimiter != '/') {
        return false;  // 不是有效的 RTSP URL 格式
    }

    url_info.url = "rtsp://";
    std::streampos pos = iss.tellg();
    std::string str = rtsp_url.substr(static_cast<int>(pos));
    if (str.find('@') != std::string::npos){
        std::getline(iss, url_info.username, ':');  // 获取用户名
        std::getline(iss, url_info.password, '@');  // 获取密码
    }
    
    pos = iss.tellg();
    str = rtsp_url.substr(static_cast<int>(pos));
    url_info.url += str;
    if(str.find(':') != std::string::npos){
        std::getline(iss, url_info.host, ':');  // 获取主机地址
        if (url_info.host.empty()) {
            return false;  // 主机地址不能为空
        }

        std::string port;
        std::getline(iss, port, ':');  // 端口
        url_info.port = atoi(port.c_str());
    }
    else if(str.find('/') != std::string::npos){
        std::getline(iss, url_info.host, '/');  // 获取主机地址
        url_info.port = 554; // 默认端口号为 554
    }
    else{
        url_info.host = str;
        url_info.port = 554; // 默认端口号为 554
    }
    return true;
}
static bool ParseRTSPLine(const std::string& line, std::string& key, std::string& value) {
    size_t pos = line.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    key = line.substr(0, pos);
    
    // 跳过冒号后的所有空格
    size_t start = pos + 1;
    while (start < line.size() && std::isspace(line[start])) {
        ++start;
    }
    value = line.substr(start);

    return true;
}
char *GetLineFromBuf(char *buf, char *line, int buf_len)
{
    while ((buf_len > 0) && (*buf != '\n'))
    {
        *line = *buf;
        line++;
        buf++;
        buf_len--;
    }

    *line = '\n';
    ++line;
    *line = '\0';
    if(buf_len > 0){
        ++buf;
    }
    return buf;
}
int ParseRTSPMessage(const std::string& rtsp_message, struct ResponseMessage &response) {
    int used = 0; 
    std::vector<std::pair<std::string, std::string>> result;
    char line[1024];
    char *buffer_end = const_cast<char*>(rtsp_message.c_str()) + rtsp_message.size() -1;
    char *buffer_ptr = const_cast<char*>(rtsp_message.c_str());
    char state_buffer[512] = {0};
    response.code = -1;
    response.find_payload = false;
    while(buffer_ptr < buffer_end){ // 跳过上一个消息数据
        buffer_ptr = GetLineFromBuf(buffer_ptr, line, buffer_end - buffer_ptr);
        used += strlen(line);
        if (sscanf(line, "RTSP/1.0 %d %s\r\n", &response.code, state_buffer) == 2) {
            break;
        }        
    }
    while(buffer_ptr < buffer_end){
        buffer_ptr = GetLineFromBuf(buffer_ptr, line, buffer_end - buffer_ptr);
        used += strlen(line);
        if(line[0] == '\r' && line[1] == '\n'){ // have payload
            response.find_payload = true;
            break;
        }
        std::string key;
        std::string value;
        int line_len = strlen(line);
        line[line_len-2] = '\0';
        ParseRTSPLine(line, key, value);
        result.emplace_back(key, value);
    }
    response.result = result;
    response.message = state_buffer;
    return used;
}
std::string GetValueByKey(const std::vector<std::pair<std::string, std::string>>& headers, std::string key) {
    std::string lower_key = key;
    // 转换 key 为小写
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
    for (const auto& header : headers) {
        std::string lower_header = header.first;
        // 转换 header.first 为小写
        std::transform(lower_header.begin(), lower_header.end(), lower_header.begin(), ::tolower);
        if (lower_header == lower_key) {
            return header.second;
        }
    }
    return "";
}
std::string GenerateAuthResponse(const char *username, const char *password, const char *realm, const char *nonce, const char *uri, const char * method){
    std::string response;
    // md5(username:realm:password)
    unsigned char res1[16];
    char res1_hex[33] = {0};
    char buffer1[256] = {0};
    sprintf(buffer1,"%s:%s:%s", username, realm, password);
    MD5_CTX md5_1;
    MD5Init(&md5_1);
    MD5Update(&md5_1, (unsigned char*)buffer1, strlen(buffer1));
    MD5Final(&md5_1, res1);
    for(int i = 0; i < 16; i++) {
        snprintf(&(res1_hex[i * 2]), 3, "%02x", res1[i]);
    }
    // md5(public_method:url)
    unsigned char res2[16];
    char res2_hex[33] = {0};
    char buffer2[256] = {0};
    sprintf(buffer2,"%s:%s", method, uri);
    MD5_CTX md5_2;
    MD5Init(&md5_2);
    MD5Update(&md5_2, (unsigned char*)buffer2, strlen(buffer2));
    MD5Final(&md5_2, res2);
    for(int i = 0; i < 16; i++) {
        snprintf(&(res2_hex[i * 2]), 3, "%02x", res2[i]);
    }
    // md5( md5(username:realm:password):nonce:md5(public_method:url) )
    unsigned char res[16];
    char res_hex[33] = {0};
    char buffer[512] = {0};
    sprintf(buffer,"%s:%s:%s", res1_hex, nonce, res2_hex);
    MD5_CTX md5;
    MD5Init(&md5);
    MD5Update(&md5, (unsigned char*)buffer, strlen(buffer));
    MD5Final(&md5, res);
    for(int i = 0; i < 16; i++) {
        snprintf(&(res_hex[i * 2]), 3, "%02x", res[i]);
    }
    response = res_hex;
    return response;
}
int CreateRtpSockets(int *fd1, int *fd2, int *port1, int *port2)
{
    struct sockaddr_in addr;
    int port = 0;

    *fd1 = socket(AF_INET, SOCK_DGRAM, 0);
    if (*fd1 < 0)
    {
        perror("socket");
        return -1;
    }

    *fd2 = socket(AF_INET, SOCK_DGRAM, 0);
    if (*fd2 < 0)
    {
        perror("socket");
        close(*fd1);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    for (port = 1024; port <= 65535; port += 2)
    {
        addr.sin_port = htons(port);
        if (bind(*fd1, (struct sockaddr *)&addr, sizeof(addr)) == 0)
        {
            addr.sin_port = htons(port + 1);
            if (bind(*fd2, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            {
                *port1 = port;
                *port2 = port + 1;
                return 0;
            }
            close(*fd1);
        }
    }
    close(*fd1);
    close(*fd2);
    return -1;
}
