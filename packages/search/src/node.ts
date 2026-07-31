import type { Move, Position } from '@sinsan/rules';

export class MCTSNode {
  readonly position: Position;
  readonly parent: MCTSNode | null;
  /** The move that produced this node from its parent - undefined only for the root. */
  readonly moveFromParent: Move | undefined;

  visitCount = 0;
  valueSum = 0;
  expanded = false;
  terminal = false;

  /** Prior probability per legal action at this node, set once at expansion. */
  readonly priors = new Map<number, number>();
  /** The actual Move for each legal action id, set once at expansion - avoids re-deriving it
   * from packages/action-space's decodeAction (which only returns squares, not full legality)
   * every time a child is created. */
  readonly movesByAction = new Map<number, Move>();
  readonly children = new Map<number, MCTSNode>();

  constructor(position: Position, parent: MCTSNode | null, moveFromParent: Move | undefined) {
    this.position = position;
    this.parent = parent;
    this.moveFromParent = moveFromParent;
  }

  /** Mean value from THIS node's own to-move perspective (docs/MODEL_DESIGN.md's convention). */
  get meanValue(): number {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }
}
