#ifndef EUCLIDEAN_MST_CUSTOM_VOLUMES_H
#define EUCLIDEAN_MST_CUSTOM_VOLUMES_H

[[nodiscard]] long double log_unit_ball_volume(int n);
[[nodiscard]] long double log_mvgamma(int k, long double a);
[[nodiscard]] long double log_volume_O(int k);
[[nodiscard]] long double log_grassmann_volume(int n, int k);

[[nodiscard]] double grassmann_volume(int n, int k);
[[nodiscard]] double unit_ball_volume(int n);

#endif // EUCLIDEAN_MST_CUSTOM_VOLUMES_H