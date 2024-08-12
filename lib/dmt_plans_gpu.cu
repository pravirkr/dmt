#include <dmt/dmt_plans_gpu.hpp>

FDMTPlanD FDMTPlanD::create_from_plan(const FDMTPlan& plan) {
    FDMTPlanD plan_d;
    const auto& plan_c = plan.get_container();
    // Transfer the plan to the device
    const auto niter_size = static_cast<int>(plan_c.state_shape.size());

    // Temp vectors to store the flattened plan on the host
    std::vector<int> nsubs_h, ncoords_h, nsamps_h, ncoords_to_copy_h;
    std::vector<int> subs_iter_idx_h, coords_iter_idx_h,
        coords_to_copy_iter_idx_h, mappings_iter_idx_h,
        mappings_to_copy_iter_idx_h;
    std::vector<int> coordinates_h, coordinates_to_copy_h, mappings_h,
        mappings_to_copy_h, state_sub_idx_h;
    nsubs_h.reserve(niter_size);
    ncoords_h.reserve(niter_size);
    nsamps_h.reserve(niter_size);
    ncoords_to_copy_h.reserve(niter_size);
    subs_iter_idx_h.reserve(niter_size);
    coords_iter_idx_h.reserve(niter_size);
    coords_to_copy_iter_idx_h.reserve(niter_size);
    mappings_iter_idx_h.reserve(niter_size);
    mappings_to_copy_iter_idx_h.reserve(niter_size);
    for (int i = 0; i < niter_size; ++i) {
        nsubs_h.emplace_back(plan_c.state_shape[i][0]);
        nsamps_h.emplace_back(plan_c.state_shape[i][4]);
        ncoords_h.emplace_back(plan_c.coordinates[i].size());
        ncoords_to_copy_h.emplace_back(plan_c.coordinates_to_copy[i].size());
    }

    // Cumulative sum
    subs_iter_idx_h.emplace_back(0);
    coords_iter_idx_h.emplace_back(0);
    coords_to_copy_iter_idx_h.emplace_back(0);
    mappings_iter_idx_h.emplace_back(0);
    mappings_to_copy_iter_idx_h.emplace_back(0);
    for (int i = 1; i < niter_size; ++i) {
        subs_iter_idx_h.emplace_back(subs_iter_idx_h[i - 1] + nsubs_h[i - 1]);
        coords_iter_idx_h.emplace_back(coords_iter_idx_h[i - 1] +
                                       ncoords_h[i - 1] * 2);
        coords_to_copy_iter_idx_h.emplace_back(
            coords_to_copy_iter_idx_h[i - 1] + ncoords_to_copy_h[i - 1] * 2);
        mappings_iter_idx_h.emplace_back(mappings_iter_idx_h[i - 1] +
                                         ncoords_h[i - 1] * 5);
        mappings_to_copy_iter_idx_h.emplace_back(
            mappings_to_copy_iter_idx_h[i - 1] + ncoords_to_copy_h[i - 1] * 5);
    }

    // Resize the vectors
    coordinates_h.reserve(coords_iter_idx_h.back() + ncoords_h.back() * 2);
    coordinates_to_copy_h.reserve(coords_to_copy_iter_idx_h.back() +
                                  ncoords_to_copy_h.back() * 2);
    mappings_h.reserve(mappings_iter_idx_h.back() + ncoords_h.back() * 5);
    mappings_to_copy_h.reserve(mappings_to_copy_iter_idx_h.back() +
                               ncoords_to_copy_h.back() * 5);
    state_sub_idx_h.reserve(subs_iter_idx_h.back() + nsubs_h.back());
    // Flatten the coordinates and mappings
    for (int i = 0; i < niter_size; ++i) {
        for (int j = 0; j < ncoords_h[i]; ++j) {
            const auto& coord   = plan_c.coordinates[i][j];
            const auto& mapping = plan_c.mappings[i][j];
            coordinates_h.emplace_back(coord.first);
            coordinates_h.emplace_back(coord.second);
            mappings_h.emplace_back(mapping.tail.first);
            mappings_h.emplace_back(mapping.tail.second);
            mappings_h.emplace_back(mapping.head.first);
            mappings_h.emplace_back(mapping.head.second);
            mappings_h.emplace_back(mapping.offset);
        }
        for (int j = 0; j < ncoords_to_copy_h[i]; ++j) {
            const auto& coord   = plan_c.coordinates_to_copy[i][j];
            const auto& mapping = plan_c.mappings_to_copy[i][j];
            coordinates_to_copy_h.emplace_back(coord.first);
            coordinates_to_copy_h.emplace_back(coord.second);
            mappings_to_copy_h.emplace_back(mapping.tail.first);
            mappings_to_copy_h.emplace_back(mapping.tail.second);
            mappings_to_copy_h.emplace_back(mapping.head.first);
            mappings_to_copy_h.emplace_back(mapping.head.second);
            mappings_to_copy_h.emplace_back(mapping.offset);
        }
        for (int j = 0; j < nsubs_h[i]; ++j) {
            state_sub_idx_h.emplace_back(plan_c.state_sub_idx[i][j]);
        }
    }

    // Copy to device
    nsubs_d                     = nsubs_h;
    nsamps_d                    = nsamps_h;
    ncoords_d                   = ncoords_h;
    ncoords_to_copy_d           = ncoords_to_copy_h;
    subs_iter_idx_d             = subs_iter_idx_h;
    coords_iter_idx_d           = coords_iter_idx_h;
    coords_to_copy_iter_idx_d   = coords_to_copy_iter_idx_h;
    mappings_iter_idx_d         = mappings_iter_idx_h;
    mappings_to_copy_iter_idx_d = mappings_to_copy_iter_idx_h;
    coordinates_d               = coordinates_h;
    coordinates_to_copy_d       = coordinates_to_copy_h;
    mappings_d                  = mappings_h;
    mappings_to_copy_d          = mappings_to_copy_h;
    state_sub_idx_d             = state_sub_idx_h;

    // dt_grid allocation for initialisation
    const auto& dt_grid_init = plan_c.dt_grid[0];

    std::vector<int> ndt_grid_init_h, dt_grid_init_sub_idx_h, dt_grid_init_h;
    ndt_grid_init_h.reserve(dt_grid_init.size());
    for (const auto& dt_grid : dt_grid_init) {
        ndt_grid_init_h.emplace_back(dt_grid.size());
    }
    dt_grid_init_sub_idx_h.emplace_back(0);
    for (size_t i = 1; i < ndt_grid_init_h.size(); ++i) {
        dt_grid_init_sub_idx_h.emplace_back(dt_grid_init_sub_idx_h[i - 1] +
                                            ndt_grid_init_h[i - 1]);
    }
    dt_grid_init_h.reserve(dt_grid_init_sub_idx_h.back() +
                           ndt_grid_init_h.back());
    for (const auto& dt_grid : dt_grid_init) {
        for (const auto& dt : dt_grid) {
            dt_grid_init_h.emplace_back(dt);
        }
    }
    ndt_grid_init_d        = ndt_grid_init_h;
    dt_grid_init_sub_idx_d = dt_grid_init_sub_idx_h;
    dt_grid_init_d         = dt_grid_init_h;
}

FDMTPlanD FDMTPlanD::create_from_plan2(const FDMTPlan& plan) {
    FDMTPlanD device_plan;
    const auto& plan_c = plan.get_container();
    const int niter_size = static_cast<int>(plan_c.state_shape.size());

    // Extract and flatten data from plan
    auto extract = [&](auto getter) {
        return flatten(std::vector<std::vector<int>>(niter_size, getter(plan_c)));
    };

    device_plan.nsubs_d = extract([](const auto& p) { return p.state_shape[0]; });
    device_plan.nsamps_d = extract([](const auto& p) { return p.state_shape[4]; });
    device_plan.ncoords_d = extract([](const auto& p) { return std::vector<int>{static_cast<int>(p.coordinates.size())}; });
    device_plan.ncoords_to_copy_d = extract([](const auto& p) { return std::vector<int>{static_cast<int>(p.coordinates_to_copy.size())}; });

    // Compute cumulative sums
    device_plan.subs_iter_idx_d = cumulative_sum(std::vector<int>(device_plan.nsubs.begin(), device_plan.nsubs.end()));
    device_plan.coords_iter_idx_d = cumulative_sum(std::vector<int>(device_plan.ncoords.begin(), device_plan.ncoords.end()));
    device_plan.coords_to_copy_iter_idx_d = cumulative_sum(std::vector<int>(device_plan.ncoords_to_copy.begin(), device_plan.ncoords_to_copy.end()));
    device_plan.mappings_iter_idx_d = cumulative_sum(std::vector<int>(device_plan.ncoords.begin(), device_plan.ncoords.end()));
    device_plan.mappings_to_copy_iter_idx_d = cumulative_sum(std::vector<int>(device_plan.ncoords_to_copy.begin(), device_plan.ncoords_to_copy.end()));

    // Flatten coordinates and mappings
    auto flatten_coord_map = [](const auto& coords, const auto& maps) {
        std::vector<int> flat_coords, flat_maps;
        for (size_t i = 0; i < coords.size(); ++i) {
            for (const auto& coord : coords[i]) {
                flat_coords.push_back(coord.first);
                flat_coords.push_back(coord.second);
            }
            for (const auto& map : maps[i]) {
                flat_maps.push_back(map.tail.first);
                flat_maps.push_back(map.tail.second);
                flat_maps.push_back(map.head.first);
                flat_maps.push_back(map.head.second);
                flat_maps.push_back(map.offset);
            }
        }
        return std::make_pair(flat_coords, flat_maps);
    };

    auto [flat_coords, flat_maps] = flatten_coord_map(plan_c.coordinates, plan_c.mappings);
    auto [flat_coords_to_copy, flat_maps_to_copy] = flatten_coord_map(plan_c.coordinates_to_copy, plan_c.mappings_to_copy);

    device_plan.coordinates_d = flat_coords;
    device_plan.coordinates_to_copy_d = flat_coords_to_copy;
    device_plan.mappings_d = flat_maps;
    device_plan.mappings_to_copy_d = flat_maps_to_copy;

    device_plan.state_sub_idx_d = flatten(plan_c.state_sub_idx);

    // Handle dt_grid initialization
    const auto& dt_grid_init_host = plan_c.dt_grid[0];
    device_plan.ndt_grid_init_d = std::vector<int>(dt_grid_init_host.size());
    std::transform(dt_grid_init_host.begin(), dt_grid_init_host.end(), device_plan.ndt_grid_init.begin(),
                    [](const auto& grid) { return static_cast<int>(grid.size()); });

    device_plan.dt_grid_init_sub_idx_d = cumulative_sum(std::vector<int>(device_plan.ndt_grid_init.begin(), device_plan.ndt_grid_init.end()));
    device_plan.dt_grid_init_d = flatten(dt_grid_init_host);

    return device_plan;
}