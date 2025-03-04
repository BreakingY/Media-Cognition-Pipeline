#ifndef _RTSP_COMMON_H_
#define _RTSP_COMMON_H_

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <string.h>

#define RTP_VESION      2
#define RTP_HEADER_SIZE 12

/*
 *
 *    0                   1                   2                   3
 *    0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7
 *   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *   |V=2|P|X|  CC   |M|     PT      |       sequence number         |
 *   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *   |                           timestamp                           |
 *   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *   |           synchronization source (SSRC) identifier            |
 *   +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
 *   |            contributing source (CSRC) identifiers             |
 *   :                             ....                              :
 *   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 */
struct RtpHeader
{
    /* byte 0 */
    uint8_t csrcLen : 4;
    uint8_t extension : 1;
    uint8_t padding : 1;
    uint8_t version : 2;

    /* byte 1 */
    uint8_t payloadType : 7;
    uint8_t marker : 1;

    /* bytes 2,3 */
    uint16_t seq;

    /* bytes 4-7 */
    uint32_t timestamp;

    /* bytes 8-11 */
    uint32_t ssrc;
};

struct RtpPacket
{
    struct RtpHeader rtpHeader;
    uint8_t payload[0];
};
#if 0
struct rtp_tcp_header
{
    uint8_t magic;   // $
    uint8_t channel; // 0-1
    uint16_t rtp_len16;
};
#endif
struct rtp_tcp_header
{
    uint8_t magic;   // $
    int channel; // 0-1
    int rtp_len16;
};
struct RTSPUrlInfo {
    std::string username;
    std::string password;
    std::string url;      // URL after removing username and password
    std::string host;
    int port;
};
struct ResponseMessage {
    int code;
    std::string message;  // Status description
    std::vector<std::pair<std::string, std::string>> result; // headers
    std::string sdp; // Describe response
    bool find_payload;
};
enum MediaEnum{
    H264 = 0,   // H264
    H265,       // H265
    AAC,        // MPEG4-GENERIC
    PCMA,       // PCMA
    NONE,
};
bool ParseRTSPUrl(const std::string& rtsp_url, RTSPUrlInfo& url_info);
char *GetLineFromBuf(char *buf, char *line, int buf_len);
int ParseRTSPMessage(const std::string& rtsp_message, struct ResponseMessage &response);
std::string GetValueByKey(const std::vector<std::pair<std::string, std::string>>& headers, std::string key);
std::string GenerateAuthResponse(const char *username, const char *password, const char *realm, const char *nonce, const char *uri, const char * method);
#endif
