from datetime import datetime, timedelta

import pytest

from p2p_node import (
    Node,
    NodeStatus,
    P2PMessage,
    P2PNetwork,
    MessageType,
    P2P_NODE_TIMEOUT,
)


def test_node_authenticity_and_liveness():
    node = Node(node_id="n1", host="localhost", port=8000)
    assert node.is_authentic is True
    assert node.is_alive is True

    node.last_seen = datetime.utcnow() - timedelta(seconds=P2P_NODE_TIMEOUT + 5)
    assert node.is_alive is False

    node.update_last_seen()
    assert node.status == NodeStatus.ALIVE


def test_message_sign_and_verify():
    message = P2PMessage(
        message_id="m1",
        message_type=MessageType.HEARTBEAT,
        from_node_id="local",
        to_node_id=None,
        payload={"ping": 1},
    )

    assert message.verify() is False

    message.sign(secret="secret")
    assert message.verify(secret="secret") is True
    assert message.verify(secret="wrong") is False


@pytest.mark.asyncio
async def test_handle_message_triggers_registered_handler():
    local = Node(node_id="local", host="127.0.0.1", port=8000)
    network = P2PNetwork(local_node=local, enable_broadcast=False, enable_registry=False)

    called = []

    def handler(msg):
        called.append(msg.message_id)

    network.register_handler(MessageType.HEARTBEAT, handler)

    incoming = P2PMessage(
        message_id="mid-1",
        message_type=MessageType.HEARTBEAT,
        from_node_id="peer",
        to_node_id=local.node_id,
        payload={"timestamp": datetime.utcnow().isoformat()},
    )
    incoming.sign()

    await network.handle_message(incoming)

    assert called == ["mid-1"]


def test_network_add_node_validates_authenticity():
    local = Node(node_id="local", host="127.0.0.1", port=8000)
    network = P2PNetwork(local_node=local, enable_broadcast=False, enable_registry=False)

    fake = Node(node_id="fake", host="1.2.3.4", port=9000, sacred_hash="123")
    assert network.add_node(fake) is False
    assert network.get_network_status()["total_nodes"] == 0

    peer = Node(node_id="peer", host="1.2.3.4", port=9000)
    assert network.add_node(peer) is True

    status = network.get_network_status()
    assert status["total_nodes"] == 1
    assert status["alive_nodes"] == 1
