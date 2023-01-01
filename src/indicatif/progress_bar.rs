use std::borrow::Cow;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

use super::draw_target::{
    ProgressDrawState, ProgressDrawTarget,
};
use super::state::{ProgressState, Status};
use super::style::ProgressStyle;
use super::{ProgressBarIter, ProgressIterator};

/// A progress bar or spinner
///
/// The progress bar is an [`Arc`] around its internal state. When the progress bar is cloned it
/// just increments the refcount (so the original and its clone share the same state).
#[derive(Clone)]
pub struct ProgressBar {
    state: Arc<Mutex<ProgressState>>,
}

impl fmt::Debug for ProgressBar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProgressBar").finish()
    }
}

impl ProgressBar {
    /// Creates a new progress bar with a given length
    ///
    /// This progress bar by default draws directly to stderr, and refreshes a maximum of 15 times
    /// a second. To change the refresh rate, set the draw target to one with a different refresh
    /// rate.
    pub fn new(len: u64) -> ProgressBar {
        ProgressBar::with_draw_target(len, ProgressDrawTarget::stderr())
    }

    /// Creates a completely hidden progress bar
    ///
    /// This progress bar still responds to API changes but it does not have a length or render in
    /// any way.
    pub fn hidden() -> ProgressBar {
        ProgressBar::with_draw_target(!0, ProgressDrawTarget::hidden())
    }

    /// Creates a new progress bar with a given length and draw target
    pub fn with_draw_target(len: u64, target: ProgressDrawTarget) -> ProgressBar {
        ProgressBar {
            state: Arc::new(Mutex::new(ProgressState::new(len, target))),
        }
    }

    /// A convenience builder-like function for a progress bar with a given style
    pub fn with_style(self, style: ProgressStyle) -> ProgressBar {
        self.state.lock().unwrap().style = style;
        self
    }

    /// A convenience builder-like function for a progress bar with a given prefix
    pub fn with_prefix(self, prefix: impl Into<Cow<'static, str>>) -> ProgressBar {
        self.state.lock().unwrap().prefix = prefix.into();
        self
    }

    /// A convenience builder-like function for a progress bar with a given message
    pub fn with_message(self, message: impl Into<Cow<'static, str>>) -> ProgressBar {
        self.state.lock().unwrap().message = message.into();
        self
    }

    /// A convenience builder-like function for a progress bar with a given position
    pub fn with_position(self, pos: u64) -> ProgressBar {
        self.state.lock().unwrap().pos = pos;
        self
    }

    /// Creates a new spinner
    ///
    /// This spinner by default draws directly to stderr. This adds the default spinner style to it.
    pub fn new_spinner() -> ProgressBar {
        let rv = ProgressBar::new(!0);
        rv.set_style(ProgressStyle::default_spinner());
        rv
    }

    /// Overrides the stored style
    ///
    /// This does not redraw the bar. Call [`ProgressBar::tick()`] to force it.
    pub fn set_style(&self, style: ProgressStyle) {
        self.state.lock().unwrap().style = style;
    }

    /// Spawns a background thread to tick the progress bar
    ///
    /// When this is enabled a background thread will regularly tick the progress bar in the given
    /// interval (in milliseconds). This is useful to advance progress bars that are very slow by
    /// themselves.
    ///
    /// When steady ticks are enabled, calling [`ProgressBar::tick()`] on a progress bar does not
    /// have any effect.
    pub fn enable_steady_tick(&self, ms: u64) {
        let mut state = self.state.lock().unwrap();
        state.steady_tick = ms;
        if state.tick_thread.is_some() {
            return;
        }

        // Using a weak pointer is required to prevent a potential deadlock. See issue #133
        let state_arc = Arc::downgrade(&self.state);
        state.tick_thread = Some(thread::spawn(move || Self::steady_tick(state_arc, ms)));

        ::std::mem::drop(state);
        // use the side effect of tick to force the bar to tick.
        self.tick();
    }

    fn steady_tick(state_arc: Weak<Mutex<ProgressState>>, mut ms: u64) {
        loop {
            thread::sleep(Duration::from_millis(ms));
            if let Some(state_arc) = state_arc.upgrade() {
                let mut state = state_arc.lock().unwrap();
                if state.is_finished() || state.steady_tick == 0 {
                    state.steady_tick = 0;
                    state.tick_thread = None;
                    break;
                }
                if state.tick != 0 {
                    state.tick = state.tick.saturating_add(1);
                }
                ms = state.steady_tick;

                state.draw().ok();
            } else {
                break;
            }
        }
    }

    /// Undoes [`ProgressBar::enable_steady_tick()`]
    pub fn disable_steady_tick(&self) {
        self.enable_steady_tick(0);
    }

    /// Limit redrawing of progress bar to every `n` steps
    ///
    /// By default, the progress bar will redraw whenever its state advances. This setting is
    /// helpful in situations where the overhead of redrawing the progress bar dominates the
    /// computation whose progress is being reported.
    ///
    /// If `n` is greater than 0, operations that change the progress bar such as
    /// [`ProgressBar::tick()`], [`ProgressBar::set_message()`] and [`ProgressBar::set_length()`]
    /// will no longer cause the progress bar to be redrawn, and will only be shown once the
    /// position advances by `n` steps.
    /// Note that `ProgressDrawTarget` may impose additional buffering of redraws.
    pub fn set_draw_delta(&self, n: u64) {
        let mut state = self.state.lock().unwrap();
        state.draw_delta = n;
        state.draw_next = state.pos.saturating_add(state.draw_delta);
    }

    /// Sets the refresh rate of progress bar to `n` updates per seconds
    ///
    /// This is similar to `set_draw_delta` but automatically adapts to a constant refresh rate
    /// regardless of how consistent the progress is.
    ///
    /// This parameter takes precedence on `set_draw_delta` if different from 0.
    ///
    /// Note that the [`ProgressDrawTarget`] may impose additional buffering of redraws.
    pub fn set_draw_rate(&self, n: u64) {
        let mut state = self.state.lock().unwrap();
        state.draw_rate = n;
        state.draw_next = state
            .pos
            .saturating_add((state.per_sec() / n as f64) as u64);
    }

    /// Manually ticks the spinner or progress bar
    ///
    /// This automatically happens on any other change to a progress bar.
    pub fn tick(&self) {
        self.update_and_draw(|state| {
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        });
    }

    /// Advances the position of the progress bar by `delta`
    pub fn inc(&self, delta: u64) {
        self.update_and_draw(|state| {
            state.pos = state.pos.saturating_add(delta);
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
    }

    /// A quick convenience check if the progress bar is hidden
    pub fn is_hidden(&self) -> bool {
        self.state.lock().unwrap().draw_target.is_hidden()
    }

    /// Indicates that the progress bar finished
    pub fn is_finished(&self) -> bool {
        self.state.lock().unwrap().is_finished()
    }

    /// Print a log line above the progress bar
    ///
    /// If the progress bar was added to a [`MultiProgress`], the log line will be
    /// printed above all other progress bars.
    ///
    /// Note that if the progress bar is hidden (which by default happens if the progress bar is
    /// redirected into a file) `println()` will not do anything either.
    pub fn println<I: AsRef<str>>(&self, msg: I) {
        let mut state = self.state.lock().unwrap();

        let mut lines: Vec<String> = msg.as_ref().lines().map(Into::into).collect();
        let orphan_lines = lines.len();
        if state.should_render() && !state.draw_target.is_hidden() {
            lines.extend(state.style.format_state(&state));
        }

        let draw_state = ProgressDrawState {
            lines,
            orphan_lines,
            finished: state.is_finished(),
            force_draw: true,
            move_cursor: false,
        };

        state.draw_target.apply_draw_state(draw_state).ok();
    }

    /// Sets the position of the progress bar
    pub fn set_position(&self, pos: u64) {
        self.update_and_draw(|state| {
            state.draw_next = pos;
            state.pos = pos;
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
    }

    /// Sets the length of the progress bar
    pub fn set_length(&self, len: u64) {
        self.update_and_draw(|state| {
            state.len = len;
        })
    }

    /// Increase the length of the progress bar
    pub fn inc_length(&self, delta: u64) {
        self.update_and_draw(|state| {
            state.len = state.len.saturating_add(delta);
        })
    }

    /// Sets the current prefix of the progress bar
    ///
    /// For the prefix to be visible, the `{prefix}` placeholder must be present in the template
    /// (see [`ProgressStyle`]).
    pub fn set_prefix(&self, prefix: impl Into<Cow<'static, str>>) {
        let prefix = prefix.into();
        self.update_and_draw(|state| {
            state.prefix = prefix;
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
    }

    /// Sets the current message of the progress bar
    ///
    /// For the message to be visible, the `{msg}` placeholder must be present in the template (see
    /// [`ProgressStyle`]).
    pub fn set_message(&self, msg: impl Into<Cow<'static, str>>) {
        let msg = msg.into();
        self.update_and_draw(|state| {
            state.message = msg;
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
    }

    /// Creates a new weak reference to this `ProgressBar`
    pub fn downgrade(&self) -> WeakProgressBar {
        WeakProgressBar {
            state: Arc::downgrade(&self.state),
        }
    }

    /// Resets the ETA calculation
    ///
    /// This can be useful if the progress bars made a large jump or was paused for a prolonged
    /// time.
    pub fn reset_eta(&self) {
        self.update_and_draw(|state| {
            state.est.reset();
        });
    }

    /// Resets elapsed time
    pub fn reset_elapsed(&self) {
        self.update_and_draw(|state| {
            state.started = Instant::now();
        });
    }

    /// Resets all of the progress bar state
    pub fn reset(&self) {
        self.reset_eta();
        self.reset_elapsed();
        self.update_and_draw(|state| {
            state.draw_next = 0;
            state.pos = 0;
            state.status = Status::InProgress;
        });
    }

    /// Finishes the progress bar and leaves the current message
    pub fn finish(&self) {
        self.state.lock().unwrap().finish();
    }

    /// Finishes the progress bar at current position and leaves the current message
    pub fn finish_at_current_pos(&self) {
        self.state.lock().unwrap().finish_at_current_pos();
    }

    /// Finishes the progress bar and sets a message
    ///
    /// For the message to be visible, the `{msg}` placeholder must be present in the template (see
    /// [`ProgressStyle`]).
    pub fn finish_with_message(&self, msg: impl Into<Cow<'static, str>>) {
        self.state.lock().unwrap().finish_with_message(msg);
    }

    /// Finishes the progress bar and completely clears it
    pub fn finish_and_clear(&self) {
        self.state.lock().unwrap().finish_and_clear();
    }

    /// Finishes the progress bar and leaves the current message and progress
    pub fn abandon(&self) {
        self.state.lock().unwrap().abandon();
    }

    /// Finishes the progress bar and sets a message, and leaves the current progress
    ///
    /// For the message to be visible, the `{msg}` placeholder must be present in the template (see
    /// [`ProgressStyle`]).
    pub fn abandon_with_message(&self, msg: impl Into<Cow<'static, str>>) {
        self.state.lock().unwrap().abandon_with_message(msg);
    }

    /// Finishes the progress bar using the behavior stored in the [`ProgressStyle`]
    ///
    /// See [`ProgressStyle::on_finish()`].
    pub fn finish_using_style(&self) {
        self.state.lock().unwrap().finish_using_style();
    }

    /// Sets a different draw target for the progress bar
    ///
    /// This can be used to draw the progress bar to stderr (this is the default):
    ///
    /// **Note:** Calling this method on a [`ProgressBar`] linked with a [`MultiProgress`] (after
    /// running [`MultiProgress::add`]) will unlink this progress bar. If you don't want this
    /// behavior, call [`MultiProgress::set_draw_target`] instead.
    pub fn set_draw_target(&self, target: ProgressDrawTarget) {
        let mut state = self.state.lock().unwrap();
        state.draw_target.disconnect();
        state.draw_target = target;
    }

    /// Wraps an [`Iterator`] with the progress bar
    pub fn wrap_iter<It: Iterator>(&self, it: It) -> ProgressBarIter<It> {
        it.progress_with(self.clone())
    }

    /// Wraps an [`io::Read`] with the progress bar
    pub fn wrap_read<R: io::Read>(&self, read: R) -> ProgressBarIter<R> {
        ProgressBarIter {
            progress: self.clone(),
            it: read,
        }
    }

    /// Wraps an [`io::Write`] with the progress bar
    pub fn wrap_write<W: io::Write>(&self, write: W) -> ProgressBarIter<W> {
        ProgressBarIter {
            progress: self.clone(),
            it: write,
        }
    }

    fn update_and_draw<F: FnOnce(&mut ProgressState)>(&self, f: F) {
        // Delegate to the wrapped state.
        let mut state = self.state.lock().unwrap();
        state.update_and_draw(f);
    }

    /// Returns the current position
    pub fn position(&self) -> u64 {
        self.state.lock().unwrap().pos
    }

    /// Returns the current length
    pub fn length(&self) -> u64 {
        self.state.lock().unwrap().len
    }

    /// Returns the current ETA
    pub fn eta(&self) -> Duration {
        self.state.lock().unwrap().eta()
    }

    /// Returns the current rate of progress
    pub fn per_sec(&self) -> f64 {
        self.state.lock().unwrap().per_sec()
    }

    /// Returns the current expected duration
    pub fn duration(&self) -> Duration {
        self.state.lock().unwrap().duration()
    }

    /// Returns the current elapsed time
    pub fn elapsed(&self) -> Duration {
        self.state.lock().unwrap().started.elapsed()
    }
}

/// A weak reference to a `ProgressBar`.
///
/// Useful for creating custom steady tick implementations
#[derive(Clone)]
pub struct WeakProgressBar {
    state: Weak<Mutex<ProgressState>>,
}

impl WeakProgressBar {
    /// Attempts to upgrade the Weak pointer to a [`ProgressBar`], delaying dropping of the inner
    /// value if successful. Returns `None` if the inner value has since been dropped.
    ///
    /// [`ProgressBar`]: struct.ProgressBar.html
    pub fn upgrade(&self) -> Option<ProgressBar> {
        self.state.upgrade().map(|state| ProgressBar { state })
    }
}