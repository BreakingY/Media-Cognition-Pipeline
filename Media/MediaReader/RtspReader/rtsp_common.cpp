#include <sstream>
#include "rtsp_common.h"
extern "C"{
#include "md5.h"
}
bool ParseRTSPUrl(const std::string& rtsp_url, RTSPUrlInfo& url_info) {
    // format: rtsp://[username:password@]host[:port]
    std::istringstream iss(rtsp_url);
    char delimiter;

    // Check if it starts with "rtsp://""
    if (!(iss >> delimiter) || delimiter != 'r' ||
        !(iss >> delimiter) || delimiter != 't' ||
        !(iss >> delimiter) || delimiter != 's' ||
        !(iss >> delimiter) || delimiter != 'p' ||
        !(iss >> delimiter) || delimiter != ':' ||
        !(iss >> delimiter) || delimiter != '/' ||
        !(iss >> delimiter) || delimiter != '/') {
        return false;  // Not a valid RTSP URL format
    }

    url_info.url = "rtsp://";
    std::streampos pos = iss.tellg();
    std::string str = rtsp_url.substr(static_cast<int>(pos));
    if (str.find('@') != std::string::npos){
        std::getline(iss, url_info.username, ':');  // Get username
        std::getline(iss, url_info.password, '@');  // Get password
    }
    
    pos = iss.tellg();
    str = rtsp_url.substr(static_cast<int>(pos));
    url_info.url += str;
    if(str.find(':') != std::string::npos){
        std::getline(iss, url_info.host, ':');  // Get host address
        if (url_info.host.empty()) {
            return false;  // The host address cannot be empty
        }

        std::string port;
        std::getline(iss, port, ':');  // port
        url_info.port = atoi(port.c_str());
    }
    else if(str.find('/') != std::string::npos){
        std::getline(iss, url_info.host, '/');  // Get host address
        url_info.port = 554; // The default port number is 554
    }
    else{
        url_info.host = str;
        url_info.port = 554; // The default port number is 554
    }
    return true;
}
static bool ParseRTSPLine(const std::string& line, std::string& key, std::string& value) {
    size_t pos = line.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    key = line.substr(0, pos);
    
    // Skip all spaces after the colon
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
    while(buffer_ptr < buffer_end){ // Skip the previous message data
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
    // Convert key to lowercase
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
    for (const auto& header : headers) {
        std::string lower_header = header.first;
        // Convert header.first to lowercase
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
