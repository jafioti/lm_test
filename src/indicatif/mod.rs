//! indicatif is a library for Rust that helps you build command line
//! interfaces that report progress to users.  It comes with various
//! tools and utilities for formatting anything that indicates progress.
//!
//! Platform support:
//!
//! * Linux
//! * OS X
//! * Windows (colors require Windows 10)
//!
//! Best paired with other libraries in the family:
//!
//! * [console](https://docs.rs/console)
//! * [dialoguer](https://docs.rs/dialoguer)
//!
//! # Crate Contents
//!
//! * **Progress bars**
//!   * [`ProgressBar`](struct.ProgressBar.html) for bars and spinners
//!   * [`MultiProgress`](struct.MultiProgress.html) for multiple bars
//! * **Data Formatting**
//!   * [`HumanBytes`](struct.HumanBytes.html) for formatting bytes
//!   * [`DecimalBytes`](struct.DecimalBytes.html) for formatting bytes using SI prefixes
//!   * [`BinaryBytes`](struct.BinaryBytes.html) for formatting bytes using ISO/IEC prefixes
//!   * [`HumanDuration`](struct.HumanDuration.html) for formatting durations
//!
//! # Progress Bars and Spinners
//!
//! indicatif comes with a `ProgressBar` type that supports both bounded
//! progress bar uses as well as unbounded "spinner" type progress reports.
//! Progress bars are `Sync` and `Send` objects which means that they are
//! internally locked and can be passed from thread to thread.
//!
//! Additionally a `MultiProgress` utility is provided that can manage
//! rendering multiple progress bars at once (eg: from multiple threads).
//!
//! To whet your appetite, this is what this can look like:
//!
//! <img src="https://github.com/mitsuhiko/indicatif/raw/main/screenshots/yarn.gif?raw=true" width="60%">
//!
//! Progress bars are manually advanced and by default draw to stderr.
//! When you are done, the progress bar can be finished either visibly
//! (eg: the progress bar stays on the screen) or cleared (the progress
//! bar will be removed).
//!
//! General progress bar behaviors:
//!
//! * if a non terminal is detected the progress bar will be completely
//!   hidden.  This makes piping programs to logfiles make sense out of
//!   the box.
//! * a progress bar only starts drawing when `set_message`, `inc`, `set_position`
//!   or `tick` are called.  In some situations you might have to call `tick`
//!   once to draw it.
//! * progress bars should be explicitly finished to reset the rendering
//!   for others.  Either by also clearing them or by replacing them with
//!   a new message / retaining the current message.
//! * the default template renders neither message nor prefix.
//!
//! # Iterators
//!
//! Similar to [tqdm](https://github.com/tqdm/tqdm), progress bars can be
//! associated with an iterator. For example:
//!
//!
//! See the [`ProgressIterator`](trait.ProgressIterator.html) trait for more
//! methods to configure the number of elements in the iterator or change
//! the progress bar style. Indicatif also has optional support for parallel
//! iterators with [Rayon](https://github.com/rayon-rs/rayon). In your
//! `Cargo.toml`, use the "rayon" feature:
//!
//! Or if you'd like to customize the progress bar:
//!
//! # Templates
//!
//! Progress bars can be styled with simple format strings similar to the
//! ones in Rust itself.  The format for a placeholder is `{key:options}`
//! where the `options` part is optional.  If provided the format is this:

//! For the style component see [`Style::from_dotted_str`](https://docs.rs/console/0.7.5/console/struct.Style.html#method.from_dotted_str)
//! for more information.  Indicatif uses the `console` base crate for all
//! colorization and formatting options.
//!
//! Some examples for templates:
//! This sets a progress bar that is 40 characters wide and has cyan
//! as primary style color and blue as alternative style color.
//! Alternative styles are currently only used for progress bars.
//!
//! Example configuration:
//! The following keys exist:
//!
//! * `bar`: renders a progress bar. By default 20 characters wide.  The
//!   style string is used to color the elapsed part, the alternative
//!   style is used for the bar that is yet to render.
//! * `wide_bar`: like `bar` but always fills the remaining space.
//! * `spinner`: renders the spinner (current tick string).
//! * `prefix`: renders the prefix set on the progress bar.
//! * `msg`: renders the currently set message on the progress bar.
//! * `wide_msg`: like `msg` but always fills the remaining space and truncates.
//! * `pos`: renders the current position of the bar as integer
//! * `len`: renders the total length of the bar as integer
//! * `bytes`: renders the current position of the bar as bytes.
//! * `percent`: renders the current position of the bar as a percentage of the total length.
//! * `total_bytes`: renders the total length of the bar as bytes.
//! * `elapsed_precise`: renders the elapsed time as `HH:MM:SS`.
//! * `elapsed`: renders the elapsed time as `42s`, `1m` etc.
//! * `per_sec`: renders the speed in steps per second.
//! * `bytes_per_sec`: renders the speed in bytes per second.
//! * `binary_bytes_per_sec`: renders the speed in bytes per second using
//!   power-of-two units, i.e. `MiB`, `KiB`, etc.
//! * `eta_precise`: the remaining time (like `elapsed_precise`).
//! * `eta`: the remaining time (like `elapsed`).
//! * `duration_precise`: the extrapolated total duration (like `elapsed_precise`).
//! * `duration`: the extrapolated total duration time (like `elapsed`).

//!
//! The design of the progress bar can be altered with the integrated
//! template functionality.  The template can be set by changing a
//! `ProgressStyle` and attaching it to the progress bar.
//!
//! # Human Readable Formatting
//!
//! There are some formatting wrappers for showing elapsed time and
//! file sizes for human users:
//!
//! # Feature Flags
//!
//! * `rayon`: adds rayon support
//! * `improved_unicode`: adds improved unicode support (graphemes, better width calculation)

mod draw_target;
mod format;
mod iter;
mod progress_bar;
#[cfg(feature = "rayon")]
mod rayon;
mod state;
mod style;

pub use draw_target::{MultiProgressAlignment, ProgressDrawTarget};
pub use format::{BinaryBytes, DecimalBytes, FormattedDuration, HumanBytes, HumanDuration};
pub use iter::{ProgressBarIter, ProgressIterator};
pub use progress_bar::{ProgressBar, WeakProgressBar};
pub use style::{ProgressFinish, ProgressStyle};

#[cfg(feature = "rayon")]
pub use super::rayon::ParallelProgressIterator;
