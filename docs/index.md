# Home {: .hidden-title }

![eventlab logo](assets/header_logo.png)

Welcome to `eventlab`! This package is a one-stop-shop for generating catalogs of stellar events from COSMIC output. The goal for this package is to include all the functions you need to compare your COSMIC runs to different types of observations. The current research focus this is being developed for are:

- Investigating the impact of binary evolution on population properties and delay time distributions of evolved stars including Wolf-Rayets, red/yellow/blue supergiants, and luminous blue variables
- Generating catalogs of CCSNe with types (I/II) and subtypes (IIP, IIL, IIb, IIn, Ib, Ic) by extending the [ccsnlab](https://github.com/MarkGM02/COSMIC-ccsnlab) package

But the applications are much wider than this! Anything that you can think of as an 'event' whether it's instantaneous, extended, related to a single star or a phase of binary interaction, this package has you covered. The main module to generate synthetic catalogs is the [events](reference/events.md) module, which is supported by a bunch of other modules to help generate useful information to put in your catalog. If you want to get a sense for some of the applications of the code check out our demo notebooks.

Additionally, each module has its own API reference.

- [events](reference/events.md)
- [dtd](reference/dtd.md)
- [imf](reference/imf.md)
- [photometry](reference/photometry.md)

We hope this package is helpful in your endeavors! This project is currently maintained by Mark Martinez, an astrophysics PhD student at the University of Pittsburgh. For any questions these docs don't answer, feel free to reach out at [mark.martinez@pitt.edu](mailto:mark.martinez@pitt.edu).