// KillSwitch is an atomic bool shared by every goroutine in the system.
// When set, all GameEngines exit within one loop iteration and no new orders
// are placed. Can be set via: config flag at startup, daily loss breach,
// or manual trigger (API endpoint or signal handler).
package risk

import "sync/atomic"

type KillSwitch struct {
	val atomic.Bool
}

func NewKillSwitch() *KillSwitch { return &KillSwitch{} }

func (ks *KillSwitch) Set()   { ks.val.Store(true) }
func (ks *KillSwitch) IsSet() bool { return ks.val.Load() }
