#ifndef _RTSP_COMMON_H_
#define _RTSP_COMMON_H_

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#define RTP_VESION 2 // 版本
#define RTP_HEADER_SIZE 12 // 头部大小

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
/*
*版本号（V）：2比特，用来标志使用的RTP版本。

*填充位（P）：1比特，如果该位置位，则该RTP包的尾部就包含附加的填充字节。

*扩展位（X）：1比特，如果该位置位的话，RTP固定头部后面就跟有一个扩展头部。

*CSRC计数器（CC）：4比特，含有固定头部后面跟着的CSRC的数目。

*标记位（M）：1比特,该位的解释由配置文档（Profile）来承担.

*载荷类型（PT）：7比特，标识了RTP载荷的类型。

*序列号（SN）：16比特，发送方在每发送完一个RTP包后就将该域的值增加1，接收方可以由该域检测包的丢失及恢复包序列。序列号的初始值是随机的。

*时间戳：32比特，记录了该包中数据的第一个字节的采样时刻。在一次会话开始时，时间戳初始化成一个初始值。即使在没有信号发送时，时间戳的数值也要随时间而不断地增加（时间在流逝嘛）。时间戳是去除抖动和实现同步不可缺少的。

*同步源标识符(SSRC)：32比特，同步源就是指RTP包流的来源。在同一个RTP会话中不能有两个相同的SSRC值。该标识符是随机选取的 RFC1889推荐了MD5随机算法。

*贡献源列表（CSRC List）：0～15项，每项32比特，用来标志对一个RTP混合器产生的新包有贡献的所有RTP包的源。由混合器将这些有贡献的SSRC标识符插入表中。SSRC标识符都被列出来，以便接收端能正确指出交谈双方的身份
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
    std::string username;  // 用户名
    std::string password;  // 密码
    std::string url;      // 去掉用户名、密码后的url
    std::string host;      // 主机地址
    int port;              // 端口号
};
struct ResponseMessage {
    int code;  // 状态码
    std::string message;  // 状态描述
    std::vector<std::pair<std::string, std::string>> result; //头部
    std::string sdp; // describe响应
    bool find_payload;
};
enum MediaEnum{
    H264 = 0,   //H264
    H265,       // H265
    AAC,        // MPEG4-GENERIC
    PCMA,       // PCMA
    NONE,
};
int CreateTcpSocket();
int CreateUdpSocket();
int ConnectToServer(int sockfd, const char* ip, int port, int timeout); // timeout:ms
int BindSocketAddr(int sockfd, const char *ip, int port);
bool ParseRTSPUrl(const std::string& rtsp_url, RTSPUrlInfo& url_info);
char *GetLineFromBuf(char *buf, char *line, int buf_len);
int ParseRTSPMessage(const std::string& rtsp_message, struct ResponseMessage &response);
std::string GetValueByKey(const std::vector<std::pair<std::string, std::string>>& headers, std::string key);
std::string GenerateAuthResponse(const char *username, const char *password, const char *realm, const char *nonce, const char *uri, const char * method);
int CreateRtpSockets(int *fd1, int *fd2, int *port1, int *port2);
#endif
