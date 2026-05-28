// worker_ipc.h — minimal length-prefixed frame protocol over a Unix-domain
// socketpair, used by longcat-avatar-server's subprocess-worker model.
//
// Shape ported (trimmed) from qwen3-tts.cpp/src/worker_ipc.h. Parent role is
// HTTP server + drain/unload state + admin endpoints, NO GPU. Worker role
// is sd_ctx + CUDA + weights, started via fork()+execv("--worker <fd>").
// Single in-flight render per worker (matches the supervisor's serialised
// semantics).
//
// Protocol: fixed 12-byte header followed by `payload_len` bytes.
//
//   [u32 frame_type][u32 payload_len][u32 req_id][u8 payload[payload_len]]
//
// RENDER_REQ payload: pack_render_payload() = [u32 json_len][json bytes]
//   [u32 image_len][image bytes][u32 audio_len][audio bytes]. JSON carries
//   the SDGenerationParams JSON + tmp-paths the worker should write to;
//   image/audio are raw container bytes (png/jpg/webm/wav).
//
// RENDER_RESP payload: [u32 json_len][json bytes][u32 video_len][video bytes]
//   where json carries {ok, error, render_sec, segments, frame_count, fps}
//   and video is the encoded container (webm/avi/webp).

#pragma once

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

namespace longcat_avatar {

enum class WorkerFrame : uint32_t {
    HELLO       = 0x01,  // W->P  {"pid": int}
    LOAD_REQ    = 0x10,  // P->W  {... ctx params JSON ...}
    LOAD_RESP   = 0x11,  // W->P  {"ok": bool, "error": str}
    RENDER_REQ  = 0x20,  // P->W  packed render payload (json+image+audio)
    RENDER_RESP = 0x21,  // W->P  packed render response (json+video)
    PING        = 0x40,
    PONG        = 0x41,
    SHUTDOWN    = 0xFF,
};

struct FrameHeader {
    uint32_t type;
    uint32_t len;
    uint32_t req_id;
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

inline constexpr size_t HEADER_BYTES = sizeof(FrameHeader);
// 512 MiB safety cap — webm for a 1-minute 480x832 clip is well under this.
inline constexpr size_t MAX_FRAME_PAYLOAD = 512ull * 1024 * 1024;

enum class IpcError {
    OK = 0,
    EofClean,
    EofMidFrame,
    SocketError,
    ProtocolError,
    PayloadTooBig,
};

const char* ipc_error_str(IpcError e);

IpcError read_exact(int fd, void* buf, size_t len);
IpcError write_exact(int fd, const void* buf, size_t len);

IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id,
                    const void* payload, size_t payload_len);
IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id,
                    const std::string& json);
IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id,
                    const std::vector<uint8_t>& payload);

IpcError recv_frame(int fd, FrameHeader* out_hdr,
                    std::vector<uint8_t>* out_payload);

// Pack JSON + image bytes + audio bytes into a single RENDER_REQ payload.
std::vector<uint8_t> pack_render_request(const std::string& json_meta,
                                         const std::vector<uint8_t>& image,
                                         const std::vector<uint8_t>& audio);
bool unpack_render_request(const std::vector<uint8_t>& payload,
                           std::string* out_meta,
                           std::vector<uint8_t>* out_image,
                           std::vector<uint8_t>* out_audio);

// Pack JSON + video bytes into a single RENDER_RESP payload.
std::vector<uint8_t> pack_render_response(const std::string& json_meta,
                                          const std::vector<uint8_t>& video);
bool unpack_render_response(const std::vector<uint8_t>& payload,
                            std::string* out_meta,
                            std::vector<uint8_t>* out_video);

// socketpair() + fork() + execv(argv[0], "--worker <fd>" extra_argv...).
// Parent gets parent-side fd in *out_parent_fd and the child pid as return.
pid_t spawn_worker(const char* self_argv0,
                   const std::vector<std::string>& extra_argv,
                   int* out_parent_fd,
                   const char* role_flag = "--worker");

} // namespace longcat_avatar
