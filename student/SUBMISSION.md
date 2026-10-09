# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phạm Văn Kiên
- MSSV: 2A202602590
- Email: 
- Link repo (fork): https://github.com/pvksssss/K4-Track4-Day23-PhamVanKien-2A202602590-Sensor-Fusion-Student
- Commit hash nộp (`git rev-parse HEAD`): ef1f7bf17dc19efffaa1aad1849bc809186279f0

## Tóm tắt kết quả

> Chạy từ root repo:
>
> ```bash
> fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
> ```
>
> với `student/config/paths.yaml`: `frame_start: 0`, `frame_end: 198` (chạy cả segment
> `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
> — segment mặc định `1005081002024129653` không có trong dữ liệu khóa học đã nhận;
> dùng segment thay thế từ cùng Google Drive Day23-Fusion).

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
  `compare`, `[0, 197]`, `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`, `seed = 0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
  `0.6455`, `0.7396`, `tp = 71`, `fp = 39`, `fn = 25`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  `0.2097` m, `matches = 45`, `sum_sq_err = 1.9779`, `ghost_track_frames = 14`, `missed_gt_frames = 51`, `mean_confirmed_tracks = 0.298`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  `0.1981` m, `matches = 45`, `sum_sq_err = 1.7663`, `ghost_track_frames = 14`, `missed_gt_frames = 51`, `mean_confirmed_tracks = 0.298`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Cả hai mode có **cùng** `matches`, `ghost_track_frames`, `missed_gt_frames` và
  `mean_confirmed_tracks` — camera chỉ tinh chỉnh trạng thái EKF của các track đã
  do LiDAR quyết định (không tạo/xoá track), đúng thiết kế track-then-fuse. RMSE
  fused (0.1981 m) nhỏ hơn lidar-only (0.2097 m) 0.0115 m: đo camera ở FRONT giúp
  thu hẹp phương sai vị trí của các xe xa/xấu hình của LiDAR, nhưng chỉ cải thiện
  nhỏ vì LiDAR đã là nguồn đo vị trí 3D chính và cổng χ² loại bớt cặp camera lệch.
  Đều dưới ngưỡng đầy đủ điểm 0.45 m và chênh lệch giữa hai mode xa dưới 0.05 m.

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   - LiDAR: `z` là tâm hộp 3D `(x, y, z)` trong hệ toạ độ cảm biến (`dim_meas = 3`);
     `R = diag(σ_lidar_x², σ_lidar_y², σ_lidar_z²) = diag(0.01, 0.01, 0.01)` m².
     `h(x)` tuyến tính nên `H` là khối xoay 3×6 (3 cột vận tốc bằng 0).
   - Camera: `z` là pixel `(u, v)` (`dim_meas = 2`); `R = diag(σ_cam_i², σ_cam_j²) = diag(25, 25)`
     (pixel²). `h(x)` phi tuyến (pinhole: `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`)
     nên `H` là Jacobian 2×6 tính theo quy tắc chuỗi (đạo hàm phép chiếu × ma trận xoay
     veh→sensor); cột vận tốc vẫn bằng 0.
   - Hệ quả: innovation `γ` và `S` có kích thước khác nhau (3 vs 2 chiều), cổng χ² dùng
     bậc tự do `df = dim_meas` tương ứng (3 vs 2) nên cùng một khoảng cách vật lý bị
     “nhìn nhận” khác nhau giữa hai cảm biến.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   - Khoảng cách Euclid không biết độ không chắc của từng track: track mới khởi tạo có
     `P` rất lớn ở các trục vận tốc (σ_p44/55/66 = 50/50/5) nên ngưỡng hợp lệ của nó
     phải rộng hơn track đã ổn định. Mahalanobis `d² = γᵀS⁻¹γ` với `S = HPHᵀ + R`
     chuẩn hoá residual theo độ lớn không chắc thực tế, nên so sánh được giữa các track
     và giữa hai cảm biến có đơn vị khác nhau (m vs pixel).
   - Cổng χ² (`χ²₀.₉₉₅(df)`) loại cặp bất khả thi (xe khác, vật thể lạ, đo nhiễu) **trước**
     khi greedy chọn cặp và trước khi EKF update kéo trạng thái; nếu bỏ gating, một
     measurement lọt lưới có thể “cướp” track hoặc làm track nhảy vị trí.
   - Kiểm tra FOV được làm **trước** Mahalanobis (`association_cost_matrix` gọi
     `meas.sensor.in_fov(track.x)` trước): điểm sau camera hoặc độ sâu `x_s ≤ 1e-6`
     không chiếu được, gọi projection/Jacobian sẽ lỗi hoặc vô nghĩa.

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**
   - **Track-then-fuse**: chỉ một danh sách track (`manager.track_list`), mỗi frame
     `EKF predict` **một lần**, sau đó update bằng LiDAR (AssocL), rồi update thêm bằng
     camera khi xe trong FOV (AssocC). Không ghép dữ liệu thô của hai cảm biến trước khi
     detect, cũng không chạy hai tracker riêng rồi hợp nhất kết quả sau (that would be
     fuse-then-track).
   - Trên luồng code (`platform/fusion_lab/scripts/run_lab.py`, một vòng lặp frame):
     `assoc.associate_and_update(manager, observations, KF, lidar_sensor)` rồi — chỉ khi
     `--fusion fused/compare` — `assoc.associate_and_update(manager, front_observations,
     KF, camera_sensor)` trên **cùng** `manager`/`KF`. Hai lượt association trên cùng
     danh sách track trong cùng một frame là dấu hiệu của track-then-fuse.
   - Trên `grade_run.log`: mode `fused` có đúng một record mỗi frame chứa kết quả sau khi
     cả hai lượt update đã chạy (không tách record “lidar update” / “camera update”);
     `matches + ghosts == confirmed` và `matches + misses == valid_gt` vẫn đúng — chứng
     minh camera chỉ tinh chỉnh trạng thái, không sinh/xoá track. So sánh
     `tracking.lidar` và `tracking.fused` trong `metrics.json` cho thấy đo camera chỉ
     thay đổi chất lượng ước lượng, không thay đổi cấu trúc tập track.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   - Lệch extrinsic (xoay/dịch) làm `h(x)` — tâm hộp mô phỏng hóa ra pixel — bị lệch có
     **hướng và có hệ thống**: `γ = z − h(x)` sẽ lệch quanh một offset ≠ 0 (bias) thay vì
     quanh 0; bias theo thành phần `u` hay `v` tuỳ chiều lệch.
   - Vì `S = HPHᵀ + R` không mô hình hưởng bias, `d² = γᵀS⁻¹γ` tăng bất thường ở xe nằm
     ở vùng bị lệch → cổng χ² loại bớt cặp (camera update mất tác dụng với những xe đó)
     hoặc, nếu lệch nhỏ hơn ngưỡng, update kéo trạng thái lệch theo hướng sai → residual
     vẫn có bias sau update và RMSE fused kém hơn thay vì tốt hơn.
   - Nói cách khác, triệu chứng nhận dạng là: residual trung bình khác 0, d² tăng, số
     cặp camera được gán giảm — đúng như bonus "phân tích calibration" mô tả.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.**
   - `sensor` là tham số bắt buộc vì frame không có detection (`meas_list` rỗng) vẫn phải
     chạy vòng đời: `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`
     được gọi **luôn**, kể cả khi không có cặp nào. Nếu suy ra sensor từ nội dung
     `meas_list`, frame rỗng sẽ không biết đang ở lượt nào → bỏ quên `manage_tracks` →
     track bị miss trong FOV không bị trừ score, track “ma” tồn tại vĩnh viễn. Với
     camera, sensor tường minh cũng phân biệt rõ lượt camera (không được score/delete/birth).
   - Chỉ LiDAR quyết định tồn tại vì LiDAR đo được vị trí 3D đầy đủ, phủ quanh xe;
     việc “không có detection” ở frame này là tín hiệu đáng tin để trừ score. Camera chỉ
     đo 2D và chỉ nhìn phía trước: xe ra khỏi FOV (rẽ, vòng qua, chặn tầm nhìn) không có
     nghĩa là xe biến mất. Nếu camera được quyền score/sinh/xoá, xe vượt khỏi khung hình
     sẽ bị miss oan và xoá oan. Vì vậy thiết kế “track-then-fuse” chọn LiDAR làm **track
     owner**, camera chỉ tinh chỉnh `x`, `P` qua `filter_obj.update`.

6. **Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.**
   - Xác nhận: mỗi hit LiDAR, `update_track_score(track, associated=True)` cộng
     `+1/window` (window = 6) và chặn trên ở 1; khi `score > confirmed_threshold` (0.8)
     → `state = "confirmed"`. Track chưa confirmed mà có hit thì chuyển `initialized →
     tentative` (hoặc lên `confirmed` nếu vượt ngưỡng).
   - Giữ confirmed sau miss: miss trong FOV chỉ `score −= 1/window`, **không** hạ trạng
     thái; một miss không tước confirmed. Track confirmed chỉ được xoá qua ngưỡng score
     riêng (`delete_threshold`), không bị hạng vì một frame miss.
   - Xoá (`should_delete_track`, chỉ xét trong lượt LiDAR): `P[0,0] > max_P` (9) hoặc
     `P[1,1] > max_P` — bất kể score; hoặc track confirmed có `score < delete_threshold`
     (0.6); hoặc track chưa confirmed có `score ≤ 0`. Các điều kiện OR với nhau.

## Bonus (không bắt buộc)

Không.

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Có — trợ lý AI (opencode) hỗ trợ viết
  và rà soát code Parts E–H, soạn báo cáo, và cài đặt môi trường chạy thử.
- Dùng cho phần nào (hàm, câu hỏi, debug): implement `kalman.py`, `association.py`,
  `camera_fusion.py`, `track_management.py`; kiểm tra lại công thức EKF/gating trong
  tài liệu đề bài; debug cài đặt môi trường (pip/editable) và chạy `pytest`.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): chạy toàn bộ
  `pytest student/tests -q` (128 passed, 0 failed, 0 xfailed); đối chiếu công thức
  `F`, `Q`, `γ`, `S`, `K` với `docs/HUONG_DAN_KY_THUAT.md` và test số
  (`test_kalman.py`, `test_tracking_contract.py`, `test_tracking_regressions.py`);
  chạy `fusion-run-lab --fusion compare --seed 0` trên segment mặc định và đối chiếu
  `metrics.json` với `grade_run.log` bằng `tools/check_submission.py`.

## Checklist nộp

- [ ] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [ ] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [ ] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
