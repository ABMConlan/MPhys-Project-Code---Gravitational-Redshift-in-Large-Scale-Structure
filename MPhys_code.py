import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# MAIN ANALYSIS FUNCS:
def makehist(kx_array, ky_array, kz_array, bins, sidelength, perturbed_points, perturbed=False):
    if not perturbed:
        print(f"Making delta-r and phi-r histograms ({perturbed_points})")
        points, omegam, H0, zcosmo = loadCosmicWebData()

    if perturbed:
        print("Making delta-r and phi-r histograms (perturbed pointset)")
        omegam, H0, zcosmo = np.load("constants.npy")
        points = perturbed_points

    print("Subsample data read in.")
    print(type(points))
    print(np.shape(points))
    dim = (bins, bins, bins)
    phi_k = np.zeros(dim, dtype=complex)

    cell_volume = (sidelength / bins) ** 3
    whole_box_volume = sidelength ** 3

    udelta_r, binedges = np.histogramdd(points, bins=bins)
    average_whole_box_deltar = np.mean(udelta_r)  # The average density over the whole sim box
    print(f"Average delta(r) = {average_whole_box_deltar}")
    delta_r = (udelta_r / average_whole_box_deltar) - 1  # Normalising density.

    delta_k = np.fft.fftn(delta_r)
    print("FFT done.")

    a = 1 / (1+zcosmo)
    H0 = 100
    consts = -(1.5 * (H0 ** 2) * omegam) / a

    def get_k_mag(kx, ky, kz):
        k_mags = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
        return k_mags

    def get_phi_k(dk, kms, c, ph):
        return c * np.divide(dk, kms, out=ph * np.ones_like(dk), where=kms != 0)

    k_mags = get_k_mag(kx_array, ky_array, kz_array)
    k_mag_sqrd = k_mags ** 2

    phi_k = get_phi_k(delta_k, k_mag_sqrd, consts, ph=0)
    print("phi_k field done.")

    # Inverse FFT our phi_k field back to r-space.
    phi_r = np.fft.ifftn(phi_k) / 3e5  # Normalising by c (km/s)

    if not perturbed:
        np.save('deltar.npy', delta_r)
        np.save('deltak.npy', delta_k)
        np.save('phir.npy', phi_r)
        np.save('phik.npy', phi_k)
        print("Unperturbed density and potential fields saved.")

    if perturbed:
        np.save('deltar_pert.npy', delta_r)
        np.save('deltak_pert.npy', delta_k)
        np.save('phir_pert.npy', phi_r)
        np.save('phik_pert.npy', phi_k)
        print("Perturbed density and potential fields saved.")

def readhist(x_array, y_array, z_array, sidelength):
    # UTILITIES:
    def datacubeImage(grid, name="unspecified", title="unspecified"):
        my_vmin = np.min(grid.real)
        my_vmax = np.max(grid.real)
        my_cmap = colormap

        if name == "Delta-r":
            my_cmap = "gnuplot"
        if name == "Phi-r":
            # original_bwr = matplotlib.cm.bwr_r
            # my_cmap = shiftedColorMap(original_bwr, midpoint=0)
            my_cmap = "bwr_r"
        fig2 = plt.figure(figsize=(fs, fs))
        ax2 = fig2.add_subplot(111)
        ax2.set_title(title)
        ax2.set_xlabel("x (Mpc)")
        ax2.set_ylabel("y (Mpc)")
        img2 = ax2.imshow(grid.real[mc],
                          origin='lower', cmap=my_cmap,
                          extent=(-slen, slen, -slen, slen),
                          vmin=my_vmin, vmax=my_vmax)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig2.colorbar(img2, cax=cax, orientation='vertical')
        plt.tight_layout()

    def get_r_mag(x, y, z):
        r_mags = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        print("Obtained r-magnitude grid.")
        return r_mags

    def percentileThresholds(grid, name, percentile_list):
        flatgrid = np.ndarray.flatten(grid)
        oned_bins = 100
        histo1d, bin_edges = np.histogram(flatgrid, bins=oned_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        my_thresholds = [np.percentile(np.ndarray.flatten(grid), i) for i in percentile_list]

        fig5 = plt.figure(figsize=(fs, fs))
        ax5 = fig5.add_subplot(111)
        ax5.plot(bin_centers, histo1d, c='b')
        ax5.set_title(f"{name} percentiles = 20, 40, 60, 80, 100")
        plt.vlines(my_thresholds, 0, np.max(histo1d), colors='k', linestyles='dotted')
        
        plt.show()

        return my_thresholds

    def power_spectra(phi_k, delta_k, consts, k_mags):
        dds = delta_k * np.conjugate(delta_k) * consts ** 2
        pps = phi_k * np.conjugate(phi_k)

        flat_pps = np.ndarray.flatten(pps)
        flat_dds = np.ndarray.flatten(dds)
        flat_k = np.ndarray.flatten(k_mags)

        # The following produces, for each grid cell i, [k_i, dds_i]:
        flat_pps_and_kmag = np.stack((flat_k, flat_pps), axis=1)
        flat_dds_and_kmag = np.stack((flat_k, flat_dds), axis=1)

        # Now we sort the array by increasing k-magnitude:
        fkpps = flat_pps_and_kmag[flat_pps_and_kmag[:, 0].argsort()]
        fkdds = flat_dds_and_kmag[flat_dds_and_kmag[:, 0].argsort()]

        unique_k, unique_k_indexs, unique_k_inverse = np.unique(fkpps[:, 0],
                                                                return_index=True,
                                                                return_inverse=True)

        # Find the mean dds and pps for each k-magnitude.
        meanpps = []
        for i in range(len(unique_k_indexs) - 1):
            meanpps.append(np.mean(fkpps[unique_k_indexs[i]:unique_k_indexs[i + 1], 1]))
        meanpps.append(np.mean(fkpps[unique_k_indexs[-1], 1]))

        meandds = []
        for i in range(len(unique_k_indexs) - 1):
            meandds.append(np.mean(fkdds[unique_k_indexs[i]:unique_k_indexs[i + 1], 1]))
        meandds.append(np.mean(fkdds[unique_k_indexs[-1], 1]))

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_title("Power spectrum")
        ax.loglog(unique_k, meandds, c='b', label='Delta(k) power spectrum')
        ax.loglog(unique_k, meanpps * (unique_k ** 4), c='r', label='Phi(k) power spectrum')
        ax.loglog(unique_k, meandds - meanpps * (unique_k ** 4), c='g', label='Difference')
        ax.legend()

        plt.show()

        return meanpps, meandds

    def thresholdCells(grid, name, thresholds):
        cells_0to20   = grid < thresholds[0]
        cells_20to40  = np.logical_and(grid > thresholds[0], grid < thresholds[1])
        cells_40to60  = np.logical_and(grid > thresholds[1], grid < thresholds[2])
        cells_60to80  = np.logical_and(grid > thresholds[2], grid < thresholds[3])
        cells_80to100 = grid > thresholds[3]

        return cells_0to20, cells_20to40, cells_40to60, cells_60to80, cells_80to100

    def createCenteredGrid(grid, index, targetcell):
        xshift = np.roll(grid, targetcell[0] - index[0], axis=0)
        yshift = np.roll(xshift, targetcell[1] - index[1], axis=1)
        zshift = np.roll(yshift, targetcell[2] - index[2], axis=2)
        return zshift

    def oneradialProfile(grid, pr):
        profiling_radius = pr
        binsize = int(pr / 10)  # Mpc
        profiling_radius_axis = np.linspace(0, profiling_radius, profiling_radius // binsize)
        bingrid = (r_mags // binsize)

        grid_binning_array = np.zeros(profiling_radius // binsize)  # An empty array for each bin total
        for thisbin in range(len(grid_binning_array)):  # for each bin...
            this_bins_vals = grid[np.where(bingrid == thisbin)].real  # make a list of all the phis in this bin...
            grid_binning_array[thisbin] = np.mean(this_bins_vals)  # ...average them

        return grid_binning_array, profiling_radius_axis

    def squish(grid, indexlist):
        composite_grid = np.zeros_like(grid)
        for cell in indexlist:
            composite_grid += createCenteredGrid(grid, cell, targetcell)

        normalised_composite = composite_grid / np.shape(indexlist)[0]
        return normalised_composite

    def bunchaplots(p1, d1, p2, d2, p3, d3, p4, d4, p5, d5, p1u, d1u, p2u, d2u, p3u, d3u, p4u, d4u, p5u, d5u):
        fig, axs = plt.subplots(5, 2)
        print_ten_images_p = False
        print_ten_images_up = False
        lp = 0.0000001
        extent = (-slen, slen, -slen, slen)
        def contourlevels(grid):
            print(np.min(grid))
            print(np.mean(grid))
            return np.linspace(np.min(grid), -0.005, 10)

        if print_ten_images_p:
            im_p1 = axs[0, 0].imshow(p1[:, mc].real, origin='lower', cmap="bwr_r", extent=extent)
            im_p2 = axs[1, 0].imshow(p2[:, mc].real, origin='lower', cmap="bwr_r", extent=extent)
            im_p3 = axs[2, 0].imshow(p3[:, mc].real, origin='lower', cmap="bwr_r", extent=extent)
            im_p4 = axs[3, 0].imshow(p4[:, mc].real, origin='lower', cmap="bwr_r", extent=extent)
            im_p5 = axs[4, 0].imshow(p5[:, mc].real, origin='lower', cmap="bwr_r", extent=extent)
            im_d1 = axs[0, 1].imshow(d1[:, mc].real, origin='lower', cmap="gnuplot", extent=extent)
            #ct_d1 = axs[0, 1].contour(d1[:,mc], contourlevels(d1), colors='k', origin='upper', extent=extent)
            im_d2 = axs[1, 1].imshow(d2[:, mc].real, origin='lower', cmap="gnuplot", extent=extent)
            im_d3 = axs[2, 1].imshow(d3[:, mc].real, origin='lower', cmap="gnuplot", extent=extent)
            im_d4 = axs[3, 1].imshow(d4[:, mc].real, origin='lower', cmap="gnuplot", extent=extent)
            im_d5 = axs[4, 1].imshow(d5[:, mc].real, origin='lower', cmap="gnuplot", extent=extent)

            # PHI PLOT PARAMS:

            plt.subplots_adjust(left=0.328, right=0.634, top=0.957, wspace=0, hspace=0.234)
            imlist = [im_p1, im_p2, im_p3, im_p4, im_p5, im_d1, im_d2, im_d3, im_d4, im_d5]
            axlist = [axs[0, 0],axs[1, 0],axs[2, 0],axs[3, 0], axs[4, 0],
                      axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1], axs[4, 1]]

            for thisax, thisim in zip(axlist, imlist):
                print(str(thisax))
                divider = make_axes_locatable(thisax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(thisim, cax=cax, orientation='vertical')


        if print_ten_images_up:
            im_p1 = axs[0, 0].imshow(p1u[:, mc].real, origin='lower', cmap="bwr_r", extent=(-slen, slen, -slen, slen))
            im_p2 = axs[1, 0].imshow(p2u[:, mc].real, origin='lower', cmap="bwr_r", extent=(-slen, slen, -slen, slen))
            im_p3 = axs[2, 0].imshow(p3u[:, mc].real, origin='lower', cmap="bwr_r", extent=(-slen, slen, -slen, slen))
            im_p4 = axs[3, 0].imshow(p4u[:, mc].real, origin='lower', cmap="bwr_r", extent=(-slen, slen, -slen, slen))
            im_p5 = axs[4, 0].imshow(p5u[:, mc].real, origin='lower', cmap="bwr_r", extent=(-slen, slen, -slen, slen))
            im_d1 = axs[0, 1].imshow(d1u[:, mc].real, origin='lower', cmap="gnuplot", extent=(-slen, slen, -slen, slen))
            im_d2 = axs[1, 1].imshow(d2u[:, mc].real, origin='lower', cmap="gnuplot", extent=(-slen, slen, -slen, slen))
            im_d3 = axs[2, 1].imshow(d3u[:, mc].real, origin='lower', cmap="gnuplot", extent=(-slen, slen, -slen, slen))
            im_d4 = axs[3, 1].imshow(d4u[:, mc].real, origin='lower', cmap="gnuplot", extent=(-slen, slen, -slen, slen))
            im_d5 = axs[4, 1].imshow(d5u[:, mc].real, origin='lower', cmap="gnuplot", extent=(-slen, slen, -slen, slen))

            # PHI PLOT PARAMS:

            plt.subplots_adjust(left=0.328, right=0.634, top=0.957, wspace=0, hspace=0.234)
            imlist = [im_p1, im_p2, im_p3, im_p4, im_p5, im_d1, im_d2, im_d3, im_d4, im_d5]
            axlist = [axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0], axs[4, 0],
                      axs[0, 1], axs[1, 1], axs[2, 1], axs[3, 1], axs[4, 1]]

            for thisax, thisim in zip(axlist, imlist):
                print(str(thisax))
                divider = make_axes_locatable(thisax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(thisim, cax=cax, orientation='vertical')
            plt.show()


        pr = 100  # Mpc

        p1_profile, pra = oneradialProfile(p1, pr)
        p2_profile = oneradialProfile(p2, pr)[0]
        p3_profile = oneradialProfile(p3, pr)[0]
        p4_profile = oneradialProfile(p4, pr)[0]
        p5_profile = oneradialProfile(p5, pr)[0]
        d1_profile = oneradialProfile(d1, pr)[0]
        d2_profile = oneradialProfile(d2, pr)[0]
        d3_profile = oneradialProfile(d3, pr)[0]
        d4_profile = oneradialProfile(d4, pr)[0]
        d5_profile = oneradialProfile(d5, pr)[0]

        p1u_profile, pra = oneradialProfile(p1u, pr)
        p2u_profile = oneradialProfile(p2u, pr)[0]
        p3u_profile = oneradialProfile(p3u, pr)[0]
        p4u_profile = oneradialProfile(p4u, pr)[0]
        p5u_profile = oneradialProfile(p5u, pr)[0]
        d1u_profile = oneradialProfile(d1u, pr)[0]
        d2u_profile = oneradialProfile(d2u, pr)[0]
        d3u_profile = oneradialProfile(d3u, pr)[0]
        d4u_profile = oneradialProfile(d4u, pr)[0]
        d5u_profile = oneradialProfile(d5u, pr)[0]

        plt.tight_layout()
        
        dh_axis = np.linspace(-500,500,101)
        radialprof = True
        if radialprof:
            fig3 = plt.figure(figsize=(fs,fs))
            axs3 = fig3.add_subplot(111)
            axs3.plot(pra, p1u_profile, c="red", label="DS1, highest potential")
            axs3.plot(pra, p2u_profile, c="orange", label="DS2")
            axs3.plot(pra, p3u_profile, c="gold", label="DS3")
            axs3.plot(pra, p4u_profile, c="lime", label="DS4")
            axs3.plot(pra, p5u_profile, c="aqua", label="DS5, lowest potential")
            axs3.set_title("Phi profiles")
            axs3.set_xlabel("r (Mpc)")
            axs3.set_ylabel("phi")
            axs3.legend()


            fig4 = plt.figure(figsize=(fs, fs))
            axs4 = fig4.add_subplot(111)
            axs4.plot(pra, d1u_profile, c="red", label="DS1, lowest density")
            axs4.plot(pra, d2u_profile, c="orange", label="DS2")
            axs4.plot(pra, d3u_profile, c="gold", label="DS3")
            axs4.plot(pra, d4u_profile, c="lime", label="DS4")
            axs4.plot(pra, d5u_profile, c="aqua", label="DS5, highest density")
            axs4.plot(pra, d1_profile, c="red", linestyle="--")
            axs4.plot(pra, d2_profile, c="orange", linestyle="--")
            axs4.plot(pra, d3_profile, c="gold", linestyle="--")
            axs4.plot(pra, d4_profile, c="lime", linestyle="--")
            axs4.plot(pra, d5_profile, c="aqua", linestyle="--")
            axs4.set_title("Delta profiles")
            axs4.set_xlabel("r (Mpc)")
            axs4.set_ylabel("delta")
            axs4.legend()
            plt.show()

            fig4 = plt.figure(figsize=(fs, fs))
            axs4 = fig4.add_subplot(111)
            axs4.plot(pra, d1u_profile, c="red", label="DS1, lowest density")
            axs4.plot(pra, d2u_profile, c="orange", label="DS2")
            axs4.plot(pra, d3u_profile, c="gold", label="DS3")
            axs4.plot(pra, d4u_profile, c="lime", label="DS4")
            axs4.plot(pra, d5u_profile, c="aqua", label="DS5, highest density")
            axs4.set_title("Delta profiles")
            axs4.set_xlabel("r (Mpc)")
            axs4.set_ylabel("delta")
            axs4.legend()

        fig4 = plt.figure(figsize=(fs, fs))
        axs4 = fig4.add_subplot(111)
        r_line = r_mags[mc][mc]
        axs4.plot(dh_axis, r_line*d1[mc][mc], c="red", label="DS1, lowest density")
        axs4.plot(dh_axis, r_line*d2[mc][mc], c="orange", label="DS2")
        axs4.plot(dh_axis, r_line*d3[mc][mc], c="gold", label="DS3")
        axs4.plot(dh_axis, r_line*d4[mc][mc], c="lime", label="DS4")
        axs4.plot(dh_axis, r_line*d5[mc][mc], c="aqua", label="DS5, highest density")
        axs4.plot(dh_axis, np.flip(r_line*d1[mc][mc]), c="red", linestyle="--", alpha=0.5)
        axs4.plot(dh_axis, np.flip(r_line*d2[mc][mc]), c="orange", linestyle="--", alpha=0.5)
        axs4.plot(dh_axis, np.flip(r_line*d3[mc][mc]), c="gold", linestyle="--", alpha=0.5)
        axs4.plot(dh_axis, np.flip(r_line*d4[mc][mc]), c="lime", linestyle="--", alpha=0.5)
        axs4.plot(dh_axis, np.flip(r_line*d5[mc][mc]), c="aqua", linestyle="--", alpha=0.5)
        plt.vlines(0,-10,10, colors="grey", linestyle="--")
        plt.xlim(-100,100)
        axs4.set_title("Delta profiles")
        axs4.set_xlabel("r (Mpc)")
        axs4.set_ylabel("delta")
        axs4.legend()
        plt.show()

    def createStacks(phi, delta_r, pert="up"):
        percentiles = [20, 40, 60, 80, 100]
        delta_thresholds = percentileThresholds(delta_r, f"Delta-r-{pert}", percentiles)
        phi_thresholds = percentileThresholds(phi, f"Phi-r{pert}", percentiles)
        delta_0to20, delta_20to40, delta_40to60, delta_60to80, delta_80up = thresholdCells(delta_r,
                                                                                           f"Delta-r-{pert}",
                                                                                           delta_thresholds)

        grid_0to20_indexes = np.argwhere(delta_0to20)
        grid_20to40_indexes = np.argwhere(delta_20to40)
        grid_40to60_indexes = np.argwhere(delta_40to60)
        grid_60to80_indexes = np.argwhere(delta_60to80)
        grid_80up_indexes = np.argwhere(delta_80up)

        # Datacube imaging for 0-20-40-60-80 percentiles (delta only).
        delta_perc_imaging = True
        if delta_perc_imaging:
            print("Imaging delta percentiles...")
            datacubeImage(delta_0to20, name="Delta_0to20", title="Delta(r), 0th to 20th percentile.")
            datacubeImage(delta_20to40, name="Delta_20to40", title="Delta(r), 20th to 40th percentile.")
            datacubeImage(delta_40to60, name="Delta_40to60", title="Delta(r), 40th to 60th percentile.")
            datacubeImage(delta_60to80, name="Delta_60to80", title="Delta(r), 60th to 80th percentile.")
            datacubeImage(delta_80up, name="Delta_80to100", title="Delta(r), 80th to 100th percentile.")
            plt.show()
        else:
            print("Skipping delta percentile imaging :(")

        delta_0to20_stack = squish(delta_r, grid_0to20_indexes)
        delta_20to40_stack = squish(delta_r, grid_20to40_indexes)
        delta_40to60_stack = squish(delta_r, grid_40to60_indexes)
        delta_60to80_stack = squish(delta_r, grid_60to80_indexes)
        delta_80up_stack = squish(delta_r, grid_80up_indexes)

        phi_0to20_stack = squish(phi, grid_0to20_indexes)
        phi_20to40_stack = squish(phi, grid_20to40_indexes)
        phi_40to60_stack = squish(phi, grid_40to60_indexes)
        phi_60to80_stack = squish(phi, grid_60to80_indexes)
        phi_80up_stack = squish(phi, grid_80up_indexes)

        np.save(f"/home/aconlan/MPHYS/STACKS/Deltar_0_20{pert}.npy", delta_0to20_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Deltar_20_40{pert}.npy", delta_20to40_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Deltar_40_60{pert}.npy", delta_40to60_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Deltar_60_80{pert}.npy", delta_60to80_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Deltar_80_100{pert}.npy", delta_80up_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Phir_0_20{pert}.npy", phi_0to20_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Phir_20_40{pert}.npy", phi_20to40_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Phir_40_60{pert}.npy", phi_40to60_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Phir_60_80{pert}.npy", phi_60to80_stack)
        np.save(f"/home/aconlan/MPHYS/STACKS/Phir_80_100{pert}.npy", phi_80up_stack)
        print(f"Saved stacks.")

    r_mags = get_r_mag(x_array, y_array, z_array)

    phi, phi_k, delta_r, delta_k = readLocalFiles()
    phi_p, delta_r_p = readLocalFilesPERT()

    mc = 50
    colormap = 'plasma'
    targetcell = [mc, mc, mc]
    print(f"True if centered: {r_mags[mc, mc, mc] == 0}")
    fs = 5
    slen = sidelength / 2
    
    datacubeImage(phi, "Phi-r", "Phi-r")
    datacubeImage(delta_r, "Delta-r", "Delta-r")
    plt.show()
    
    datacubeImage(delta_r_p - delta_r, "Delta-r", "Delta-r, difference after perturbing.")
    datacubeImage(phi_p, "Phi-r", "Phi-r")
    datacubeImage(delta_r_p, "Delta-r (perturbed)", "Delta-r(perturbed)")
    
    plt.show()


    foldername = "100sp_101bins"

    p1 = np.load(f"{foldername}/Phir_0_20p.npy")
    p2 = np.load(f"{foldername}/Phir_20_40p.npy")
    p3 = np.load(f"{foldername}/Phir_40_60p.npy")
    p4 = np.load(f"{foldername}/Phir_60_80p.npy")
    p5 = np.load(f"{foldername}/Phir_80_100p.npy")

    d1 = np.load(f"{foldername}/Deltar_0_20p.npy")
    d2 = np.load(f"{foldername}/Deltar_20_40p.npy")
    d3 = np.load(f"{foldername}/Deltar_40_60p.npy")
    d4 = np.load(f"{foldername}/Deltar_60_80p.npy")
    d5 = np.load(f"{foldername}/Deltar_80_100p.npy")

    p1u = np.load(f"{foldername}/Phir_0_20up.npy")
    p2u = np.load(f"{foldername}/Phir_20_40up.npy")
    p3u = np.load(f"{foldername}/Phir_40_60up.npy")
    p4u = np.load(f"{foldername}/Phir_60_80up.npy")
    p5u = np.load(f"{foldername}/Phir_80_100up.npy")

    d1u = np.load(f"{foldername}/Deltar_0_20up.npy")
    d2u = np.load(f"{foldername}/Deltar_20_40up.npy")
    d3u = np.load(f"{foldername}/Deltar_40_60up.npy")
    d4u = np.load(f"{foldername}/Deltar_60_80up.npy")
    d5u = np.load(f"{foldername}/Deltar_80_100up.npy")


    # not pretty but I was coming up to deadline so it had to be done.
    bunchaplots(p1, d1, p2, d2, p3, d3, p4, d4, p5, d5, p1u, d1u, p2u, d2u, p3u, d3u, p4u, d4u, p5u, d5u)


    print("Done!")

def perturb(bins, sidelength):
    print("Pointset potential perturbing...")
    points = np.load("pointset.npy")
    phi_r = np.load("phir.npy")
    omegam, H0, az = np.load("constants.npy")
    H0 = 100
    print("Arrays loaded.")
    p_hist, bin_edges = np.histogramdd(points, bins=bins)

    x_d = np.digitize(points[:, 0], bin_edges[0]) - 1
    y_d = np.digitize(points[:, 1], bin_edges[1]) - 1
    z_d = np.digitize(points[:, 2], bin_edges[2]) - 1
    points_d = np.column_stack([x_d, y_d, z_d])
    points_d[points_d == bins] = bins - 1

    perturbed_points = []
    counter = 0
    arbitrary_pert_scaling = 100
    for thispoint, thispoint_d in zip(points, points_d):
        this_perturbed_point = thispoint
        z_pert = arbitrary_pert_scaling * phi_r[thispoint_d[0], thispoint_d[1], thispoint_d[2]] / H0
        this_perturbed_point[2] += z_pert.real
        this_perturbed_point[2] = (this_perturbed_point[2] + sidelength) % sidelength
        perturbed_points.append(this_perturbed_point)

        if counter % 1000000 == 0:
            print(f"{counter} of {np.shape(points)[0]}")
        counter += 1

    #np.save("perturbed_pointset.npy", perturbed_points)

    return np.vstack(perturbed_points)

# INITIALISATION / FILE I/O STUFF:

def readSubSampData(): # Filepath to original subsample data no longer available, use loadCosmicWebData() instead.
    myx = []
    myy = []
    myz = []
    head1 = []
    #fp = "/disk11/salam/MDPL2/DM/Subsamp/snap_098_0.001000.txt"
    fp = "deprecated.txt"
    with open(fp) as f:
        for line in f:
            if line.startswith("#"):
                print(line)
                if line.startswith("#Omm"):
                    omegam = float(line.split()[1])
                elif line.startswith("#h"):
                    H0 = 100 * float(line.split()[1])
                elif line.startswith("#z"):
                    az = float(line.split()[1])
                else:
                    continue
            else:
                head1.append(line)
                break
        print("Header processed")
        rest = [next(f) for line in f]
    objlist = head1 + rest
    print("Object list obtained")
    print(f"Parameters:  Omega_m={omegam}, H0={H0}, a_z={az}")
    for line in objlist:
        myx.append(float(line.split()[0]))
        myy.append(float(line.split()[1]))
        myz.append(float(line.split()[2]))

    print("Coords done")
    ps = np.column_stack([myx, myy, myz])

    print("Point set generated.")
    # np.save('pointset.npy', ps)
    # np.save('constants.npy', [omegam, H0, az])
    return ps, omegam, H0, az

def loadCosmicWebData():
    ps = np.load('pointset.npy')
    omegam, H0, az = np.load('constants.npy')
    print("Point set and constants loaded.")
    H0 = 100
    np.save("constants.npy", [omegam, H0, az])
    return ps, omegam, H0, az

def readLocalFiles():
    phi_r = np.load("phir.npy")
    phi_k = np.load("phik.npy")
    delta_r = np.load("deltar.npy")
    delta_k = np.load("deltak.npy")
    return phi_r, phi_k, delta_r, delta_k

def readLocalFilesPERT():
    phi_r = np.load("phir_pert.npy")
    #phi_k = np.load("phik_pert.npy")
    delta_r = np.load("deltar_pert.npy")
    #delta_k = np.load("deltak_pert.npy")
    return phi_r, delta_r

def rmeshgrid(bins, sidelength):
    hlen = 0.5 * sidelength
    oneaxis = np.linspace(-hlen, hlen, num=bins)
    x_axis, y_axis, z_axis = np.meshgrid(oneaxis, oneaxis, oneaxis)
    return x_axis, y_axis, z_axis

def kmeshgrid(bins, sidelength):
    n_value = np.fft.fftfreq(bins, (1.0 / bins))
    onekaxis = (2 * np.pi * n_value) / sidelength
    kx_mesh, ky_mesh, kz_mesh = np.meshgrid(onekaxis, onekaxis, onekaxis)

    return kx_mesh, ky_mesh, kz_mesh

def main():
    bins = 101
    if bins % 2 == 0:
        bins += 1 
    sidelength = 1000

    x_array, y_array, z_array = rmeshgrid(bins, sidelength)
    kx_array, ky_array, kz_array = kmeshgrid(bins, sidelength)
    print("r and k spaces initialised.")

    perturbed_points = "unperturbed pointset"

    makehist(kx_array, ky_array, kz_array, bins, sidelength, perturbed_points, perturbed=False)

    perturbed_points = perturb(bins, sidelength)

    makehist(kx_array, ky_array, kz_array, bins, sidelength, perturbed_points, perturbed=True)

    readhist(x_array, y_array, z_array, sidelength)

main()

