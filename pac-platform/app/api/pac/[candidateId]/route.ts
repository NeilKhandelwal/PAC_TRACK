import { NextResponse } from 'next/server';
import { db } from '../../../../lib/firebase';
import { doc, getDoc } from 'firebase/firestore';

export async function GET(request: Request, { params }: { params: Promise<{ candidateId: string }> }) {
    const { candidateId } = await params;
    try {
        // Assuming a collection named 'pac_spending' where each document id is candidateId
        const docRef = doc(db, 'pac_spending', candidateId);
        const snap = await getDoc(docRef);
        if (!snap.exists()) {
            return NextResponse.json({ error: 'No PAC data found' }, { status: 404 });
        }
        const data = snap.data();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Error fetching PAC data:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
