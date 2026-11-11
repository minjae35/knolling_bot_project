"""Row placement with one shared Y and a width-conditioned X step.

Each row has one neural origin. The origin head reads that row and the rows
above and below it: each row contributes its pooled features and its object
sizes, summed in row order. Later X positions add a neural step from the
previous object. That step sees only the previous width, the current width,
and the goal gap. Target poses, teacher rows, and teacher slots are training
labels. Inference passes the predicted row and the object sizes.
"""
import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.coordinates import angle_delta, wrap_pi


class NeuralRowPlacement(nn.Module):
    def __init__(self, hidden=128, goal_dim=13, max_rows=8, max_slots=8):
        super().__init__()
        if hidden < 1 or goal_dim != 13 or max_rows < 1 or max_slots < 1:
            raise ValueError('row placement width is invalid')
        self.hidden_dim = hidden
        self.max_rows = max_rows
        self.max_slots = max_slots
        self.size_features = 4
        self.row_feature_dim = hidden+self.size_features
        self.own_height_channel = hidden+2
        self.above_height_channel = self.row_feature_dim+self.own_height_channel
        width = self.row_feature_dim*3+goal_dim+max_rows
        self.row_layout = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.step = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.angle = nn.Sequential(nn.Linear(hidden+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.assign = nn.Sequential(nn.Linear(hidden+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, max_rows))
        self.order_query = nn.Sequential(
            nn.Linear(hidden*2+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.order_begin = nn.Parameter(torch.zeros(hidden))
        nn.init.zeros_(self.row_layout[-1].weight)
        self.row_layout[-1].bias.data.copy_(torch.tensor([0.3, 0.2]))
        nn.init.zeros_(self.step[-1].weight)
        nn.init.constant_(self.step[-1].bias, -2.5)

    def coordinate_parameters(self):
        return list(self.row_layout.parameters())+list(self.step.parameters())+list(self.angle.parameters())

    def origin_parameters(self):
        return list(self.row_layout.parameters())

    def selector_parameters(self):
        return list(self.assign.parameters())+list(self.order_query.parameters())+[self.order_begin]

    def origin_inputs(self, hidden, valid, goal, row, widths, heights, slot):
        """One vector per row: its own features, then the rows above, then the rows below."""
        if widths.shape != valid.shape or heights.shape != valid.shape or slot.shape != valid.shape:
            raise ValueError('row sizes do not match the objects')
        batch = len(hidden)
        table = []
        for index in range(self.max_rows):
            member = valid & (row == index)
            occupied = member.any(-1, keepdim=True)
            pooled = (hidden*member[..., None]).sum(1)/member.sum(-1, keepdim=True).clamp_min(1).to(hidden.dtype)
            pooled = torch.where(occupied, pooled, torch.zeros_like(pooled))
            count = member.sum(-1, keepdim=True).to(hidden.dtype)
            sum_width = (widths*member).sum(-1, keepdim=True)
            max_height = torch.where(member, heights, torch.zeros_like(heights)).amax(-1, keepdim=True)
            first_width = (widths*(member & (slot == 0))).sum(-1, keepdim=True)
            table.append(torch.cat((pooled, count, sum_width, max_height, first_width), -1))
        table = torch.stack(table, 1)
        parts = []
        for index in range(self.max_rows):
            code = F.one_hot(torch.full((batch,), index, device=hidden.device), self.max_rows).to(hidden.dtype)
            parts.append(torch.cat((
                table[:, index], table[:, :index].sum(1), table[:, index+1:].sum(1), goal, code), -1))
        return torch.stack(parts, 1)

    def _origins(self, hidden, valid, goal, row, widths, heights, slot):
        return self.row_layout(self.origin_inputs(hidden, valid, goal, row, widths, heights, slot))

    def _gap(self, goal):
        values = goal.new_tensor((.015, .020, .030))
        return (goal[:, 6:9]*values).sum(-1)

    def _positions(self, valid, goal, row, slot, widths, origins):
        batch, count = slot.shape
        index = row.clamp_min(0)
        origin = origins.gather(1, index[..., None].expand(-1, -1, 2))
        y = origin[..., 1]
        x = origin[..., 0]
        running = origins[..., 0]
        delta = widths.new_zeros(batch, count)
        gap = self._gap(goal)[:, None].expand(batch, count)
        for step_id in range(1, self.max_slots):
            current = valid & (slot == step_id)
            previous = valid & (slot == step_id-1)
            if not bool(current.any()):
                continue
            prev_width = widths.new_zeros(batch, self.max_rows)
            prev_width = prev_width.scatter_add(1, index, widths*previous)
            raw = self.step(torch.stack((prev_width.gather(1, index), widths, gap), -1)).squeeze(-1)
            move = F.softplus(raw)*current
            delta = torch.where(current, move, delta)
            move_row = widths.new_zeros(batch, self.max_rows)
            running = running+move_row.scatter_add(1, index, move)
            x = torch.where(current, running.gather(1, index), x)
        xy = torch.stack((x, y), -1)
        xy = torch.where(valid[..., None], xy, torch.zeros_like(xy))
        return xy, torch.where(valid, delta, torch.zeros_like(delta))

    def _order(self, hidden, valid, goal, row, teacher_slot):
        batch, count, width = hidden.shape
        index = torch.arange(batch, device=hidden.device)
        slot = torch.full((batch, count), -1, dtype=torch.long, device=hidden.device)
        logits, targets = [], []
        for row_id in range(self.max_rows):
            remaining = valid & (row == row_id)
            previous = self.order_begin.expand(batch, -1)
            for step in range(count):
                if not bool(remaining.any()):
                    break
                context = (hidden*remaining[..., None]).sum(1)/remaining.sum(-1, keepdim=True).clamp_min(1)
                query = self.order_query(torch.cat((context, previous, goal), -1))
                scores = (hidden*query[:, None]).sum(-1)/(width**.5)
                scores = scores.masked_fill(~remaining, -1e9)
                if teacher_slot is None:
                    has = remaining.any(-1)
                    picked = scores.argmax(-1)
                else:
                    wanted = remaining & (teacher_slot == step)
                    has = wanted.any(-1)
                    picked = wanted.to(hidden.dtype).argmax(-1)
                if not bool(has.any()):
                    break
                logits.append(scores)
                targets.append(torch.where(has, picked, torch.full_like(picked, -100)))
                slot[index[has], picked[has]] = step
                previous = torch.where(has[:, None], hidden[index, picked], previous)
                remaining = remaining & ~(F.one_hot(picked, count).bool() & has[:, None])
        if not logits:
            return slot, hidden.new_zeros(1, batch, count), torch.full((1, batch), -100, device=hidden.device)
        return slot, torch.stack(logits), torch.stack(targets)

    def forward(self, hidden, valid, goal, widths, heights, teacher_row=None, teacher_slot=None, predict_order=False):
        if hidden.shape[-1] != self.hidden_dim or goal.shape[-1] != 13:
            raise ValueError('row placement received the wrong hidden or goal width')
        if hidden.shape[:2] != valid.shape or widths.shape[:2] != valid.shape or len(hidden) != len(goal):
            raise ValueError('row placement batch shape mismatch')
        if heights.shape != widths.shape:
            raise ValueError('row heights do not match widths')
        if teacher_slot is None and not predict_order:
            raise ValueError('within-row placement needs a teacher slot or a predicted order')
        batch, count, _ = hidden.shape
        if count > self.max_slots:
            raise ValueError('row placement received more objects than its slot table')
        row_logits = self.assign(torch.cat((hidden, goal[:, None].expand(-1, count, -1)), -1))
        if teacher_row is None:
            row = torch.where(valid, row_logits.argmax(-1), torch.full((batch, count), -1, device=hidden.device))
        else:
            if bool((valid & (teacher_row >= self.max_rows)).any()):
                raise ValueError('teacher row exceeds the row table')
            row = torch.where(valid, teacher_row, torch.full_like(teacher_row, -1))
        order_logits = order_targets = None
        if predict_order:
            predicted, order_logits, order_targets = self._order(hidden, valid, goal, row, teacher_slot)
            slot = predicted if teacher_slot is None else torch.where(valid, teacher_slot, torch.full_like(teacher_slot, -1))
        else:
            slot = torch.where(valid, teacher_slot, torch.full_like(teacher_slot, -1))
        origins = self._origins(hidden, valid, goal, row, widths, heights, slot)
        xy, delta_x = self._positions(valid, goal, row, slot, widths, origins)
        raw_angle = self.angle(torch.cat((hidden, goal[:, None].expand(-1, count, -1)), -1))
        angle = wrap_pi(.5*torch.atan2(raw_angle[..., 1], raw_angle[..., 0])).masked_fill(~valid, 0)
        return dict(xy=xy, delta_x=delta_x, angle=angle, row=row, slot=slot, origins=origins,
                    row_logits=row_logits, order_logits=order_logits, order_targets=order_targets)


def placement_loss(output, target_xy, target_angle, valid):
    """Target poses supervise the composed coordinates. They are not model inputs."""
    if not bool(valid.any()):
        raise ValueError('placement loss needs a valid object')
    xy = F.smooth_l1_loss(output['xy'][valid], target_xy[valid], beta=.010)
    angle = angle_delta(output['angle'][valid], target_angle[valid]).abs().mean()
    return dict(loss=xy+angle, xy=xy, angle=angle)


def target_step(target_xy, valid, row, slot, max_rows=8):
    """GT Δx between consecutive teacher slots. Not a model input."""
    batch, count = slot.shape
    if int(row.max()) >= max_rows:
        raise ValueError('row index exceeds the step target table')
    index = row.clamp_min(0)
    x = target_xy[..., 0]
    previous_x = x.new_zeros(batch, count)
    covered = torch.zeros_like(valid)
    for step_id in range(1, int(slot.max())+1):
        current = valid & (slot == step_id)
        previous = valid & (slot == step_id-1)
        if not bool(current.any()):
            continue
        present = x.new_zeros(batch, max_rows)
        present = present.scatter_add(1, index, previous.to(x.dtype))
        if bool((current & (present.gather(1, index) != 1)).any()):
            raise ValueError('a stepped object has no unique predecessor in its row')
        acc = x.new_zeros(batch, max_rows)
        acc = acc.scatter_add(1, index, x*previous)
        previous_x = torch.where(current, acc.gather(1, index), previous_x)
        covered = covered | current
    return (x-previous_x).masked_fill(~covered, 0), covered


def step_placement_loss(output, target_xy, target_angle, valid, row, slot):
    """Origin X and shared Y stay absolute. Later X is supervised only as Δx.

    The rejected offset loss subtracts the same previous x from both sides, so
    it is absolute X. This loss reads the step output before it is added to
    the origin, and the later absolute X is not a target.
    """
    if not bool(valid.any()) or not bool((valid & (slot == 0)).any()):
        raise ValueError('step placement needs a row origin')
    origin = valid & (slot == 0)
    x = F.smooth_l1_loss(output['xy'][..., 0][origin], target_xy[..., 0][origin], beta=.010)
    y = F.smooth_l1_loss(output['xy'][..., 1][valid], target_xy[..., 1][valid], beta=.010)
    angle = angle_delta(output['angle'][valid], target_angle[valid]).abs().mean()
    target_dx, stepped = target_step(target_xy, valid, row, slot)
    if bool(stepped.any()):
        dx = F.smooth_l1_loss(output['delta_x'][stepped], target_dx[stepped], beta=.010)
    else:
        dx = output['xy'].new_zeros(())
    return dict(loss=x+y+dx+angle, x=x, y=y, dx=dx, angle=angle)


def origin_loss(output, target_xy, valid, slot):
    """Slot-0 pose supervises the row origin. Later X and the step are not targets."""
    chosen = valid & (slot == 0)
    if not bool(chosen.any()):
        raise ValueError('origin loss needs a row origin')
    return F.smooth_l1_loss(output['xy'][chosen], target_xy[chosen], beta=.010)


def assignment_loss(output, teacher_row, valid):
    active = valid & (teacher_row >= 0)
    if not bool(active.any()):
        raise ValueError('assignment loss needs a labeled object')
    return F.cross_entropy(output['row_logits'][active], teacher_row[active])


def order_loss(output):
    targets = output['order_targets']
    if targets is None or not bool((targets >= 0).any()):
        raise ValueError('order loss needs a teacher-forced pointer')
    return F.cross_entropy(output['order_logits'].flatten(0, 1), targets.flatten(), ignore_index=-100)
